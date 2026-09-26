"""Schema-Driven Field & Table Extraction Module - Phase 4 (Gemini Native).

Provides LLM-driven document classification, schema-constrained field and table
extraction using Google's Gemini API (google-genai), spatial bounding box grounding
back to Phase 2 OCR coordinates, and strict schema compliance.
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import re
from typing import Any

from dotenv import load_dotenv

# Ensure .env in the project directory is loaded
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_PROJECT_DIR, ".env"))

try:
    from google import genai
    from google.genai import errors as genai_errors
    GEMINI_INSTALLED = True
except ImportError:
    genai = None
    genai_errors = None
    GEMINI_INSTALLED = False

logger = logging.getLogger(__name__)

import time

# Primary & Fallback Models for Google Gemini
GEMINI_MODELS: list[str] = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-3.7-flash",
]

# Allowed document classification types
VALID_DOC_TYPES = ["invoice", "payslip", "id_proof", "bank_statement", "other"]

# Document Type Field & Table Schemas
DOC_TYPE_SCHEMAS: dict[str, dict[str, Any]] = {
    "invoice": {
        "fields": {
            "vendor_name": "name",
            "invoice_number": "id_number",
            "invoice_date": "date",
            "total_amount": "amount",
            "tax_amount": "amount",
            "currency": "text",
        },
        "tables": {
            "line_items": ["description", "quantity", "unit_price", "amount"]
        },
    },
    "payslip": {
        "fields": {
            "employee_name": "name",
            "employee_id": "id_number",
            "pay_period": "date",
            "gross_pay": "amount",
            "net_pay": "amount",
        },
        "tables": {
            "deductions": ["deduction_type", "amount"]
        },
    },
    "id_proof": {
        "fields": {
            "full_name": "name",
            "date_of_birth": "date",
            "id_number": "id_number",
            "address": "text",
            "issue_date": "date",
            "expiry_date": "date",
        },
        "tables": {},
    },
    "bank_statement": {
        "fields": {
            "account_holder_name": "name",
            "account_number": "id_number",
            "statement_period": "date",
            "closing_balance": "amount",
        },
        "tables": {
            "transactions": ["date", "description", "amount", "balance"]
        },
    },
    "other": {
        "fields": {},
        "tables": {},
    },
}


def get_gemini_client(api_key: str | None = None) -> Any:
    """Retrieve or initialize the Google GenAI client for Gemini.

    Args:
        api_key: Optional explicit API key. If omitted, reads GEMINI_API_KEY from environment/.env.

    Returns:
        genai.Client instance.

    Raises:
        RuntimeError: If google-genai is missing or GEMINI_API_KEY is not set.
    """
    if not GEMINI_INSTALLED or genai is None:
        raise RuntimeError(
            "The 'google-genai' library is not installed. Run `pip install google-genai`."
        )

    key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. Please ensure your Gemini API key is set in .env "
            "or provided in the sidebar controls."
        )

    return genai.Client(api_key=key)


def _generate_with_gemini(
    prompt: str,
    client: Any = None,
) -> str:
    """Execute generation with Gemini, gracefully handling model fallbacks."""
    if client is None:
        client = get_gemini_client()

    last_error = None
    for model_name in GEMINI_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            last_error = e
            logger.warning(f"Gemini model {model_name} failed: {e}. Trying fallback...")
            time.sleep(0.5)
            continue

    raise RuntimeError(f"All Gemini models failed. Last error: {last_error}")


def _clean_and_parse_json(text: str) -> dict[str, Any] | None:
    """Robustly extract and parse JSON from LLM response text."""
    if not text or not isinstance(text, str):
        return None

    cleaned = text.strip()

    # 1. Strip markdown code fences if present (e.g. ```json ... ``` or ``` ...)
    cleaned = re.sub(r"^```(?:json|JSON)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    # 2. Safety net: extract only substring between first "{" and last "}"
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        cleaned = cleaned[first_brace : last_brace + 1].strip()

    try:
        data = json.loads(cleaned)
        print(f"[DEBUG _clean_and_parse_json] Successfully parsed JSON. Type: {type(data)}")

        # Safety net: check for double-encoding where json.loads returns a string
        if isinstance(data, str):
            try:
                data = json.loads(data)
                print(f"[DEBUG _clean_and_parse_json] Decoded double-encoded JSON string. New type: {type(data)}")
            except Exception:
                pass

        if isinstance(data, dict):
            # Check if inner 'fields' or 'tables' are double-encoded strings
            if isinstance(data.get("fields"), str):
                try:
                    data["fields"] = json.loads(data["fields"])
                except Exception:
                    pass
            if isinstance(data.get("tables"), str):
                try:
                    data["tables"] = json.loads(data["tables"])
                except Exception:
                    pass
            return data
    except Exception as e:
        print(f"[DEBUG _clean_and_parse_json] JSON parse error: {e}")

    return None


def _infer_field_type(field_name: str, value: Any) -> str:
    """Infer field_type token (text | date | amount | id_number | name)."""
    fn = field_name.lower()
    val_str = str(value).lower()

    if any(k in fn for k in ["date", "dob", "period", "expiry", "issue"]):
        return "date"
    if any(k in fn for k in ["amount", "pay", "balance", "price", "total", "tax", "fee", "cost"]):
        return "amount"
    if any(k in fn for k in ["name", "holder", "employee", "vendor", "customer"]):
        return "name"
    if any(k in fn for k in ["number", "id", "num", "code", "account", "ssn", "ein"]):
        return "id_number"

    if re.search(r"^\$?\s*-?\d+(?:,\d{3})*(?:\.\d{1,2})?\s*(?:usd|eur|gbp|inr)?$", val_str):
        return "amount"
    if re.search(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}$", val_str):
        return "date"

    return "text"


def _is_bbox_within(inner: dict[str, int], outer: dict[str, int]) -> bool:
    """Check if inner bbox roughly lies within outer bbox vertically."""
    i_y = inner.get("y", 0)
    o_y = outer.get("y", 0)
    o_h = outer.get("height", 0)
    return (o_y - 10) <= i_y <= (o_y + o_h + 10)


def classify_doc_type(
    ocr_full_text: str,
    client: Any = None,
) -> str:
    """Classify document text into one of: invoice, payslip, id_proof, bank_statement, other.

    Args:
        ocr_full_text: Full concatenated OCR text of the document.
        client: Optional preconfigured Gemini client.

    Returns:
        One of the 5 canonical doc_type strings.
    """
    if not ocr_full_text or not ocr_full_text.strip():
        return "other"

    if client is None:
        client = get_gemini_client()

    prompt = (
        "Classify the following document OCR text into exactly one of these five categories:\n"
        "- invoice\n"
        "- payslip\n"
        "- id_proof\n"
        "- bank_statement\n"
        "- other\n\n"
        "Output ONLY the single word category name in lowercase, with no punctuation or explanation.\n\n"
        f"Document Text:\n{ocr_full_text[:4000]}"
    )

    try:
        raw_output = _generate_with_gemini(prompt=prompt, client=client)
        raw_label = raw_output.strip().lower()
        cleaned_label = re.sub(r"[^a-z_]", "", raw_label)
        if cleaned_label in VALID_DOC_TYPES:
            return cleaned_label
        return "other"
    except Exception as e:
        logger.error(f"Error classifying document type with Gemini: {e}", exc_info=True)
        raise


def build_extraction_prompt(doc_type: str, ocr_full_text: str) -> str:
    """Construct a rigorous JSON-only extraction prompt based on document schema.

    Args:
        doc_type: Document classification category.
        ocr_full_text: OCR text across all pages.

    Returns:
        Structured prompt string for Gemini.
    """
    schema = DOC_TYPE_SCHEMAS.get(doc_type, DOC_TYPE_SCHEMAS["other"])
    field_spec = list(schema["fields"].keys())
    table_spec = schema.get("tables", {})

    if doc_type == "other":
        field_instruction = "Extract any clearly labeled key-value pairs you identify in the text, using concise lowercase field names."
        table_instruction = "Extract any tabular data found in the document."
    else:
        field_instruction = (
            f"You MUST extract this exact set of fields for a {doc_type}:\n"
            + "\n".join(f"- {f}" for f in field_spec)
        )
        if table_spec:
            table_instruction = (
                f"You MUST extract the following table(s) if present in the document:\n"
                + "\n".join(f"- Table '{t_name}' with columns: {cols}" for t_name, cols in table_spec.items())
            )
        else:
            table_instruction = "No tables are expected for this document type. Return \"tables\": [] unless an explicit table exists."

    return (
        f"You are an expert Document Intelligence Extraction Engine.\n"
        f"Document Type: {doc_type}\n\n"
        f"### Target Fields:\n{field_instruction}\n\n"
        f"### Target Tables:\n{table_instruction}\n\n"
        "### STRICT INSTRUCTIONS:\n"
        "1. If a field cannot be found in the text, its value MUST be null. Never guess, infer, or fabricate a value.\n"
        "2. For currency or amounts, preserve the numerical value and extract currency into the currency field if present.\n"
        "3. Output MUST be ONLY a single valid JSON object. No explanation text, no markdown backticks, no preamble.\n"
        "4. Follow this exact JSON structure:\n"
        "{\n"
        '  "fields": {\n'
        '    "<field_name>": "<extracted value or null>"\n'
        "  },\n"
        '  "tables": [\n'
        "    {\n"
        '      "table_name": "<table_name>",\n'
        '      "rows": [\n'
        '        { "<column_name>": "<cell_value>" }\n'
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"### OCR EXTRACTED DOCUMENT TEXT:\n{ocr_full_text}"
    )


def call_llm_for_extraction(
    prompt: str,
    client: Any = None,
) -> dict[str, Any]:
    """Execute Gemini extraction call and parse JSON output with single retry.

    Args:
        prompt: Structured extraction prompt.
        client: Optional preconfigured Gemini client.

    Returns:
        Parsed dict with "fields" and "tables".
    """
    if client is None:
        client = get_gemini_client()

    try:
        content_text = _generate_with_gemini(prompt=prompt, client=client)
        print("\n" + "=" * 60)
        print("[DEBUG call_llm_for_extraction] RAW STRING RESPONSE FROM LLM BEFORE JSON PARSING:")
        print(content_text)
        print("=" * 60 + "\n")

        parsed = _clean_and_parse_json(content_text)
        if parsed is not None:
            print(f"[DEBUG call_llm_for_extraction] Parsed Python object type: {type(parsed)} (dict check: {isinstance(parsed, dict)})")
            return parsed

        logger.warning("First Gemini JSON parse failed. Retrying with strict JSON instruction...")

        # Retry once asking for strict JSON
        retry_prompt = (
            f"{prompt}\n\n"
            "CRITICAL: Your last response was not valid JSON. "
            "Respond with ONLY the valid raw JSON object, without markdown code fences or conversational text."
        )
        retry_text = _generate_with_gemini(prompt=retry_prompt, client=client)
        print("\n" + "=" * 60)
        print("[DEBUG call_llm_for_extraction] RAW STRING RESPONSE ON RETRY:")
        print(retry_text)
        print("=" * 60 + "\n")

        parsed_retry = _clean_and_parse_json(retry_text)
        if parsed_retry is not None:
            print(f"[DEBUG call_llm_for_extraction] Retry parsed Python object type: {type(parsed_retry)} (dict check: {isinstance(parsed_retry, dict)})")
            return parsed_retry

        logger.error("Gemini extraction JSON parsing failed after retry.")
        return {"fields": {}, "tables": [], "parse_error": True}

    except Exception as e:
        logger.error(f"Gemini API call failed during field extraction: {e}", exc_info=True)
        raise


def map_value_to_bbox(
    value: Any,
    ocr_results: dict[str, Any],
) -> dict[str, Any] | None:
    """Locate the spatial bounding box and raw OCR text matching an extracted value.

    Searches across all pages in ocr_results using line text, word tokens, and
    fuzzy sequence matching. Does not fabricate bounding boxes.

    Args:
        value: Extracted field or table cell value (string or number).
        ocr_results: Complete Phase 2 ocr_results dictionary.

    Returns:
        Dict with "raw_ocr_text" and "bbox" ({page, x, y, width, height}), or None if unmapped.
    """
    if value is None:
        return None

    val_str = str(value).strip()
    if not val_str or val_str.lower() in ("null", "none"):
        return None

    pages = ocr_results.get("pages", [])
    if not pages:
        return None

    val_norm = re.sub(r"[^\w]", "", val_str.lower())
    if not val_norm:
        return None

    best_match: dict[str, Any] | None = None
    best_score: float = 0.0

    for page in pages:
        page_num = page.get("page_number", 1)
        lines = page.get("lines", [])
        words = page.get("words", [])

        # 1. Line substring and exact normalized match
        for line in lines:
            line_txt = line.get("text", "")
            line_norm = re.sub(r"[^\w]", "", line_txt.lower())

            if val_str.lower() in line_txt.lower() or val_norm in line_norm:
                score = (len(val_norm) / max(len(line_norm), 1)) + 1.0
                if score > best_score:
                    matched_words = [
                        w for w in words
                        if re.sub(r"[^\w]", "", w.get("text", "").lower()) in val_norm
                        and _is_bbox_within(w.get("bbox", {}), line.get("bbox", {}))
                    ]
                    if matched_words:
                        min_x = min(w["bbox"]["x"] for w in matched_words)
                        min_y = min(w["bbox"]["y"] for w in matched_words)
                        max_x = max(w["bbox"]["x"] + w["bbox"]["width"] for w in matched_words)
                        max_y = max(w["bbox"]["y"] + w["bbox"]["height"] for w in matched_words)
                        chosen_bbox = {
                            "page": page_num,
                            "x": min_x,
                            "y": min_y,
                            "width": max_x - min_x,
                            "height": max_y - min_y,
                        }
                        raw_text = " ".join(w.get("text", "") for w in matched_words)
                    else:
                        l_b = line.get("bbox", {})
                        chosen_bbox = {
                            "page": page_num,
                            "x": l_b.get("x", 0),
                            "y": l_b.get("y", 0),
                            "width": l_b.get("width", 0),
                            "height": l_b.get("height", 0),
                        }
                        raw_text = line_txt

                    best_score = score
                    best_match = {
                        "raw_ocr_text": raw_text,
                        "bbox": chosen_bbox,
                    }

        # 2. Individual word match
        if best_score < 1.0:
            for w in words:
                w_txt = w.get("text", "")
                w_norm = re.sub(r"[^\w]", "", w_txt.lower())
                if w_norm == val_norm:
                    w_b = w.get("bbox", {})
                    best_score = 1.0
                    best_match = {
                        "raw_ocr_text": w_txt,
                        "bbox": {
                            "page": page_num,
                            "x": w_b.get("x", 0),
                            "y": w_b.get("y", 0),
                            "width": w_b.get("width", 0),
                            "height": w_b.get("height", 0),
                        },
                    }
                    break

        # 3. Fuzzy match fallback
        if best_score < 0.75:
            for line in lines:
                line_txt = line.get("text", "")
                ratio = difflib.SequenceMatcher(None, val_str.lower(), line_txt.lower()).ratio()
                if ratio > best_score and ratio >= 0.75:
                    l_b = line.get("bbox", {})
                    best_score = ratio
                    best_match = {
                        "raw_ocr_text": line_txt,
                        "bbox": {
                            "page": page_num,
                            "x": l_b.get("x", 0),
                            "y": l_b.get("y", 0),
                            "width": l_b.get("width", 0),
                            "height": l_b.get("height", 0),
                        },
                    }

    return best_match


def extract_fields(
    ocr_results: dict[str, Any],
    doc_type: str,
    client: Any = None,
) -> dict[str, Any]:
    """Extract fields and tables, mapping bounding boxes and adhering to Phase 4 schema.

    Excludes any fields where Gemini returned null.

    Args:
        ocr_results: Phase 2 ocr_results dict.
        doc_type: Document classification label.
        client: Optional preconfigured Gemini client.

    Returns:
        Dict matching Phase 4 schema.
    """
    pages = ocr_results.get("pages", [])
    ocr_full_text = "\n\n".join(
        f"--- Page {p.get('page_number', i + 1)} ---\n{p.get('full_text', '')}"
        for i, p in enumerate(pages)
    )

    prompt = build_extraction_prompt(doc_type, ocr_full_text)
    raw_extraction = call_llm_for_extraction(prompt, client=client)

    raw_fields = raw_extraction.get("fields", {})
    raw_tables = raw_extraction.get("tables", [])

    schema_config = DOC_TYPE_SCHEMAS.get(doc_type, DOC_TYPE_SCHEMAS["other"])
    expected_field_types = schema_config.get("fields", {})

    extracted_fields: dict[str, Any] = {}
    skipped_fields: list[str] = []
    unmapped_bbox_fields: list[str] = []

    for field_name, value in raw_fields.items():
        if value is None or str(value).strip().lower() in ("null", "none", ""):
            skipped_fields.append(field_name)
            continue

        field_type = expected_field_types.get(field_name) or _infer_field_type(field_name, value)
        bbox_info = map_value_to_bbox(value, ocr_results)

        if bbox_info and bbox_info.get("bbox"):
            raw_text = bbox_info["raw_ocr_text"]
            bbox = bbox_info["bbox"]
        else:
            raw_text = str(value)
            bbox = None
            unmapped_bbox_fields.append(field_name)

        extracted_fields[field_name] = {
            "value": value,
            "raw_ocr_text": raw_text,
            "bbox": bbox,
            "field_type": field_type,
        }

    # Format tables conforming to schema
    tables_output: list[dict[str, Any]] = []
    for table_idx, t in enumerate(raw_tables):
        t_name = t.get("table_name", f"table_{table_idx + 1}")
        raw_rows = t.get("rows", [])
        formatted_rows: list[dict[str, Any]] = []

        table_page = 1
        all_cell_bboxes: list[dict[str, int]] = []

        for r_idx, r_dict in enumerate(raw_rows):
            cells_dict: dict[str, Any] = {}
            for col_name, cell_val in r_dict.items():
                if cell_val is None:
                    continue
                cell_bbox_info = map_value_to_bbox(cell_val, ocr_results)
                if cell_bbox_info and cell_bbox_info.get("bbox"):
                    c_bbox = cell_bbox_info["bbox"]
                    table_page = c_bbox.get("page", 1)
                    clean_cell_bbox = {
                        "x": c_bbox["x"],
                        "y": c_bbox["y"],
                        "width": c_bbox["width"],
                        "height": c_bbox["height"],
                    }
                    all_cell_bboxes.append(clean_cell_bbox)
                else:
                    clean_cell_bbox = None

                cells_dict[col_name] = {
                    "value": cell_val,
                    "bbox": clean_cell_bbox,
                }

            formatted_rows.append({
                "row_index": r_idx + 1,
                "cells": cells_dict,
            })

        if all_cell_bboxes:
            t_min_x = min(b["x"] for b in all_cell_bboxes)
            t_min_y = min(b["y"] for b in all_cell_bboxes)
            t_max_x = max(b["x"] + b["width"] for b in all_cell_bboxes)
            t_max_y = max(b["y"] + b["height"] for b in all_cell_bboxes)
            table_bbox = {
                "page": table_page,
                "x": t_min_x,
                "y": t_min_y,
                "width": t_max_x - t_min_x,
                "height": t_max_y - t_min_y,
            }
        else:
            table_bbox = None

        tables_output.append({
            "table_name": t_name,
            "bbox": table_bbox,
            "rows": formatted_rows,
        })

    # Double-encoding / Type safety verification
    assert isinstance(extracted_fields, dict), f"extracted_fields must be dict, got {type(extracted_fields)}"
    assert isinstance(tables_output, list), f"tables must be list, got {type(tables_output)}"

    print("\n" + "=" * 60)
    print(f"[DEBUG extract_fields] FINAL EXTRACTED FIELDS (Type: {type(extracted_fields)}, Count: {len(extracted_fields)}):")
    for fn, fv in extracted_fields.items():
        print(f"  - {fn}: val={repr(fv.get('value'))} (val_type: {type(fv.get('value')).__name__}), raw_ocr={repr(fv.get('raw_ocr_text'))}, bbox={fv.get('bbox')}")
    print(f"[DEBUG extract_fields] FINAL TABLES (Type: {type(tables_output)}, Count: {len(tables_output)}):")
    for tbl in tables_output:
        print(f"  - Table: {tbl.get('table_name')}, rows: {len(tbl.get('rows', []))}")
    print("=" * 60 + "\n")

    return {
        "doc_type": doc_type,
        "extracted_fields": extracted_fields,
        "tables": tables_output,
        "_skipped_fields": skipped_fields,
        "_unmapped_bbox_fields": unmapped_bbox_fields,
    }


def run_extraction_pipeline(
    ocr_results: dict[str, Any],
    client: Any = None,
) -> dict[str, Any]:
    """Execute classification and schema-driven field extraction via Gemini.

    Args:
        ocr_results: Output of Phase 2 run_ocr_on_document.
        client: Optional preconfigured Gemini client.

    Returns:
        Dict matching Phase 4 schema contract.
    """
    if client is None:
        client = get_gemini_client()

    pages = ocr_results.get("pages", [])
    ocr_full_text = "\n\n".join(
        f"--- Page {p.get('page_number', i + 1)} ---\n{p.get('full_text', '')}"
        for i, p in enumerate(pages)
    )

    doc_type = classify_doc_type(ocr_full_text, client=client)
    return extract_fields(ocr_results, doc_type=doc_type, client=client)


# ==============================================================================
# Phase 8: Correction Memory & Adaptive Extraction
# ==============================================================================

def build_extraction_prompt_with_memory(doc_type: str, ocr_full_text: str) -> str:
    """Construct an extraction prompt augmented with few-shot reviewer correction memory.

    Args:
        doc_type: Document classification category.
        ocr_full_text: OCR text across all pages.

    Returns:
        Augmented prompt string if corrections exist, or unchanged base prompt.
    """
    import corrections

    base_prompt = build_extraction_prompt(doc_type, ocr_full_text)
    fewshot_block = corrections.build_fewshot_examples_block(doc_type)

    if not fewshot_block:
        return base_prompt

    memory_section = (
        f"\n\n### Prior correction feedback:\n"
        f"{fewshot_block}\n\n"
    )

    if "### STRICT INSTRUCTIONS:" in base_prompt:
        return base_prompt.replace(
            "### STRICT INSTRUCTIONS:",
            f"{memory_section}### STRICT INSTRUCTIONS:",
            1,
        )

    return base_prompt + memory_section


def extract_fields_with_memory(
    ocr_results: dict[str, Any],
    doc_type: str,
    client: Any = None,
) -> dict[str, Any]:
    """Extract fields using memory-augmented prompt and map coordinates.

    Args:
        ocr_results: Phase 2 ocr_results dict.
        doc_type: Document classification label.
        client: Optional preconfigured Gemini client.

    Returns:
        Dict matching Phase 4 schema contract.
    """
    pages = ocr_results.get("pages", [])
    ocr_full_text = "\n\n".join(
        f"--- Page {p.get('page_number', i + 1)} ---\n{p.get('full_text', '')}"
        for i, p in enumerate(pages)
    )

    prompt = build_extraction_prompt_with_memory(doc_type, ocr_full_text)

    # Console print showing the final prompt sent to LLM for visual confirmation
    print("\n" + "=" * 60)
    print(f"[DEBUG extraction with memory] FINAL EXTRACTION PROMPT SENT TO LLM ({doc_type}):")
    print(prompt)
    print("=" * 60 + "\n")

    raw_extraction = call_llm_for_extraction(prompt, client=client)

    raw_fields = raw_extraction.get("fields", {})
    raw_tables = raw_extraction.get("tables", [])

    schema_config = DOC_TYPE_SCHEMAS.get(doc_type, DOC_TYPE_SCHEMAS["other"])
    expected_field_types = schema_config.get("fields", {})

    extracted_fields: dict[str, Any] = {}
    skipped_fields: list[str] = []
    unmapped_bbox_fields: list[str] = []

    for field_name, value in raw_fields.items():
        if value is None or str(value).strip().lower() in ("null", "none", ""):
            skipped_fields.append(field_name)
            continue

        field_type = expected_field_types.get(field_name) or _infer_field_type(field_name, value)
        bbox_info = map_value_to_bbox(value, ocr_results)

        if bbox_info and bbox_info.get("bbox"):
            raw_text = bbox_info["raw_ocr_text"]
            bbox = bbox_info["bbox"]
        else:
            raw_text = str(value)
            bbox = None
            unmapped_bbox_fields.append(field_name)

        extracted_fields[field_name] = {
            "value": value,
            "raw_ocr_text": raw_text,
            "bbox": bbox,
            "field_type": field_type,
        }

    # Format tables conforming to schema
    tables_output: list[dict[str, Any]] = []
    for table_idx, t in enumerate(raw_tables):
        t_name = t.get("table_name", f"table_{table_idx + 1}")
        raw_rows = t.get("rows", [])
        formatted_rows: list[dict[str, Any]] = []

        table_page = 1
        all_cell_bboxes: list[dict[str, int]] = []

        for r_idx, r_dict in enumerate(raw_rows):
            cells_dict: dict[str, Any] = {}
            for col_name, cell_val in r_dict.items():
                if cell_val is None:
                    continue
                cell_bbox_info = map_value_to_bbox(cell_val, ocr_results)
                if cell_bbox_info and cell_bbox_info.get("bbox"):
                    c_bbox = cell_bbox_info["bbox"]
                    table_page = c_bbox.get("page", 1)
                    clean_cell_bbox = {
                        "x": c_bbox["x"],
                        "y": c_bbox["y"],
                        "width": c_bbox["width"],
                        "height": c_bbox["height"],
                    }
                    all_cell_bboxes.append(clean_cell_bbox)
                else:
                    clean_cell_bbox = None

                cells_dict[col_name] = {
                    "value": cell_val,
                    "bbox": clean_cell_bbox,
                }

            formatted_rows.append({
                "row_index": r_idx + 1,
                "cells": cells_dict,
            })

        if all_cell_bboxes:
            t_min_x = min(b["x"] for b in all_cell_bboxes)
            t_min_y = min(b["y"] for b in all_cell_bboxes)
            t_max_x = max(b["x"] + b["width"] for b in all_cell_bboxes)
            t_max_y = max(b["y"] + b["height"] for b in all_cell_bboxes)
            table_bbox = {
                "page": table_page,
                "x": t_min_x,
                "y": t_min_y,
                "width": t_max_x - t_min_x,
                "height": t_max_y - t_min_y,
            }
        else:
            table_bbox = None

        tables_output.append({
            "table_name": t_name,
            "bbox": table_bbox,
            "rows": formatted_rows,
        })

    return {
        "doc_type": doc_type,
        "extracted_fields": extracted_fields,
        "tables": tables_output,
        "_skipped_fields": skipped_fields,
        "_unmapped_bbox_fields": unmapped_bbox_fields,
    }


def run_extraction_pipeline_with_memory(
    ocr_results: dict[str, Any],
    client: Any = None,
) -> dict[str, Any]:
    """Execute classification and schema-driven field extraction with correction memory.

    Args:
        ocr_results: Output of Phase 2 run_ocr_on_document.
        client: Optional preconfigured Gemini client.

    Returns:
        Dict matching Phase 4 schema contract.
    """
    if client is None:
        client = get_gemini_client()

    pages = ocr_results.get("pages", [])
    ocr_full_text = "\n\n".join(
        f"--- Page {p.get('page_number', i + 1)} ---\n{p.get('full_text', '')}"
        for i, p in enumerate(pages)
    )

    doc_type = classify_doc_type(ocr_full_text, client=client)
    return extract_fields_with_memory(ocr_results, doc_type=doc_type, client=client)

