"""Per-Field Confidence Scoring Module - Phase 5.

Provides multi-factor confidence scoring:
- Spatial OCR token overlap confidence (40% weight)
- LLM self-reported certainty scoring via Gemini (40% weight)
- Deterministic regex/format rule checks (20% weight, with 0.50 cap on failure)
- Green / Amber / Red categorization and table cell scoring.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from typing import Any

from dotenv import load_dotenv

# Ensure .env is loaded
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_PROJECT_DIR, ".env"))

import extraction

logger = logging.getLogger(__name__)

# ==============================================================================
# Named Thresholds & Weights
# ==============================================================================
CONFIDENCE_GREEN_THRESHOLD: float = 0.75
CONFIDENCE_AMBER_THRESHOLD: float = 0.40

WEIGHT_OCR: float = 0.40
WEIGHT_LLM: float = 0.40
WEIGHT_RULE: float = 0.20


def _bboxes_overlap(box1: dict[str, Any], box2: dict[str, Any], margin: int = 3) -> bool:
    """Check if two 2D bounding boxes overlap spatially with a small margin."""
    x1, y1 = box1.get("x", 0) - margin, box1.get("y", 0) - margin
    w1, h1 = box1.get("width", 0) + (2 * margin), box1.get("height", 0) + (2 * margin)

    x2, y2 = box2.get("x", 0), box2.get("y", 0)
    w2, h2 = box2.get("width", 0), box2.get("height", 0)

    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)


def get_ocr_confidence_for_field(
    field_bbox: dict[str, Any] | None,
    ocr_results: dict[str, Any],
) -> float:
    """Compute average OCR confidence of all recognized words overlapping the field's bbox.

    Args:
        field_bbox: Dict with {page, x, y, width, height} or None.
        ocr_results: Phase 2 ocr_results dictionary.

    Returns:
        Average OCR confidence float in range [0.0, 1.0]. Returns 0.0 if unmapped or no words overlap.
    """
    if not field_bbox or not isinstance(field_bbox, dict):
        logger.info("Field bounding box is None or invalid. OCR confidence set to 0.0.")
        return 0.0

    target_page_num = field_bbox.get("page", 1)
    pages = ocr_results.get("pages", [])

    target_page = next((p for p in pages if p.get("page_number") == target_page_num), None)
    if not target_page:
        logger.warning(f"Target page {target_page_num} not found in OCR pages. Returning 0.0.")
        return 0.0

    words = target_page.get("words", [])
    overlapping_confidences: list[float] = []

    for w in words:
        w_bbox = w.get("bbox")
        if w_bbox and _bboxes_overlap(field_bbox, w_bbox):
            conf = float(w.get("confidence", 0.0))
            overlapping_confidences.append(conf)

    if not overlapping_confidences:
        logger.info(f"No OCR words overlap with field bbox: {field_bbox}. Returning 0.0.")
        return 0.0

    return round(float(sum(overlapping_confidences) / len(overlapping_confidences)), 4)


def get_llm_field_confidence(
    doc_type: str,
    ocr_full_text: str,
    extracted_fields: dict[str, Any],
    client: Any = None,
) -> dict[str, float]:
    """Ask Gemini to evaluate certainty (0.0 to 1.0) for each extracted field based on source text.

    Args:
        doc_type: Document classification category.
        ocr_full_text: Full OCR document text.
        extracted_fields: Dict of currently extracted fields with values.
        client: Optional preconfigured Gemini client.

    Returns:
        Dict mapping field_name to certainty float (0.0 - 1.0). Returns {} on failure.
    """
    if not extracted_fields:
        return {}

    items_to_score = {k: v.get("value") for k, v in extracted_fields.items()}

    prompt = (
        f"You are an expert document verification auditor.\n"
        f"Document Type: {doc_type}\n\n"
        "Below is the document OCR text and a set of candidate extracted fields.\n"
        "Rate your certainty (as a float between 0.0 and 1.0) for EACH extracted field based on how "
        "clearly, explicitly, and unambiguously that value appears in the document text:\n"
        "- 1.0: Exact, clear, unambiguous match in text\n"
        "- 0.7: Present with minor OCR noise or formatting variance\n"
        "- 0.4: Ambiguous or partially inferred\n"
        "- 0.0: Fabricated or absent from text\n\n"
        "Respond with ONLY a JSON object mapping each field name to its certainty score:\n"
        "{\n"
        + ",\n".join(f'  "{k}": 0.95' for k in items_to_score.keys())
        + "\n}\n\n"
        f"### FIELDS TO EVALUATE:\n{json.dumps(items_to_score, indent=2)}\n\n"
        f"### DOCUMENT OCR TEXT:\n{ocr_full_text[:4000]}"
    )

    try:
        raw_text = extraction._generate_with_gemini(prompt=prompt, client=client)
        parsed = extraction._clean_and_parse_json(raw_text)

        if isinstance(parsed, dict):
            ratings: dict[str, float] = {}
            for k in items_to_score.keys():
                val = parsed.get(k)
                if val is not None:
                    try:
                        score = float(val)
                        # Clamp between 0.0 and 1.0
                        ratings[k] = max(0.0, min(1.0, score))
                    except Exception:
                        pass
            return ratings
    except Exception as e:
        logger.warning(f"LLM field confidence scoring encountered error: {e}. Falling back to OCR + Rules.")

    return {}


def run_rule_checks(field_name: str, value: Any, field_type: str) -> bool:
    """Validate format conformance per field_type.

    Args:
        field_name: Name of the field.
        value: Extracted field value.
        field_type: Declared field type (date | amount | id_number | name | text).

    Returns:
        True if format passes validation rules, False otherwise.
    """
    if value is None:
        return False

    val_str = str(value).strip()
    if not val_str or val_str.lower() in ("null", "none"):
        return False

    f_type = field_type.lower()

    if f_type == "date":
        # Check standard date formats
        cleaned_date = re.sub(r"[,\.]", "", val_str)
        date_formats = [
            "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d",
            "%d %b %Y", "%d %B %Y", "%b %d %Y", "%B %d %Y",
            "%Y%m%d", "%d.%m.%Y", "%Y.%m.%d",
        ]
        for fmt in date_formats:
            try:
                datetime.strptime(cleaned_date, fmt)
                return True
            except ValueError:
                continue

        # Regex fallback for year/month/day patterns
        if re.search(r"\b(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\b", val_str):
            return True
        return False

    elif f_type == "amount":
        # Strip currency symbols and whitespace
        clean_amt = re.sub(r"[\$₹€£¥,\s]", "", val_str)
        # Remove currency ISO codes
        clean_amt = re.sub(r"(?i)(usd|inr|eur|gbp|cad|aud)", "", clean_amt).strip()
        try:
            float(clean_amt)
            return True
        except ValueError:
            return False

    elif f_type == "id_number":
        # Must be non-empty and have alphanumeric chars
        if len(val_str) < 2:
            return False
        return bool(re.search(r"[A-Za-z0-9]", val_str))

    elif f_type == "name":
        # Must have letters and reasonable name length
        if len(val_str) < 2 or len(val_str) > 100:
            return False
        return bool(re.search(r"[A-Za-z]", val_str))

    elif f_type == "text":
        return len(val_str) > 0

    return True


def combine_confidence(
    ocr_conf: float,
    llm_conf: float | None,
    rule_passed: bool,
) -> dict[str, Any]:
    """Compute combined confidence score and assign green/amber/red bucket.

    Formula:
    - 40% OCR confidence + 40% LLM confidence + 20% rule check passed (1.0/0.0).
    - If LLM confidence is unavailable (null), weights are renormalized across OCR (66.7%) and Rules (33.3%).
    - If rule_passed is False, combined_score is capped at 0.50 maximum.

    Buckets:
    - green: combined_score >= 0.75
    - amber: combined_score >= 0.40
    - red:   combined_score < 0.40

    Args:
        ocr_conf: Float 0-1 from OCR token overlap.
        llm_conf: Float 0-1 from LLM evaluation, or None if skipped/failed.
        rule_passed: Boolean format check result.

    Returns:
        Dict matching schema:
        {
          "ocr_confidence": float,
          "llm_confidence": float or null,
          "rule_check_passed": bool,
          "combined_score": float,
          "bucket": "green | amber | red"
        }
    """
    rule_val = 1.0 if rule_passed else 0.0

    if llm_conf is not None:
        combined = (ocr_conf * WEIGHT_OCR) + (llm_conf * WEIGHT_LLM) + (rule_val * WEIGHT_RULE)
    else:
        # Renormalize weights: OCR (0.4/0.6) and Rule (0.2/0.6)
        renorm_ocr = WEIGHT_OCR / (WEIGHT_OCR + WEIGHT_RULE)
        renorm_rule = WEIGHT_RULE / (WEIGHT_OCR + WEIGHT_RULE)
        combined = (ocr_conf * renorm_ocr) + (rule_val * renorm_rule)

    # Gate: If rule check failed, cap combined score at 0.50 maximum
    if not rule_passed:
        combined = min(combined, 0.50)

    combined = round(float(combined), 4)

    if combined >= CONFIDENCE_GREEN_THRESHOLD:
        bucket = "green"
    elif combined >= CONFIDENCE_AMBER_THRESHOLD:
        bucket = "amber"
    else:
        bucket = "red"

    return {
        "ocr_confidence": round(float(ocr_conf), 4),
        "llm_confidence": round(float(llm_conf), 4) if llm_conf is not None else None,
        "rule_check_passed": bool(rule_passed),
        "combined_score": combined,
        "bucket": bucket,
    }


def score_all_fields(
    extracted_fields: dict[str, Any],
    tables: list[dict[str, Any]],
    ocr_results: dict[str, Any],
    doc_type: str,
    ocr_full_text: str,
    client: Any = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score every extracted field and table cell, attaching the confidence sub-object.

    Args:
        extracted_fields: Dict of Phase 4 extracted fields.
        tables: List of Phase 4 extracted tables.
        ocr_results: Complete Phase 2 OCR dictionary.
        doc_type: Document classification category.
        ocr_full_text: Concatenated OCR text across pages.
        client: Optional preconfigured Gemini client.

    Returns:
        Tuple of (updated_extracted_fields, updated_tables).
    """
    # 1. Fetch LLM certainty ratings for all fields in one pass
    llm_ratings = get_llm_field_confidence(
        doc_type=doc_type,
        ocr_full_text=ocr_full_text,
        extracted_fields=extracted_fields,
        client=client,
    )

    # 2. Score each extracted field
    for field_name, finfo in extracted_fields.items():
        val = finfo.get("value")
        ftype = finfo.get("field_type", "text")
        bbox = finfo.get("bbox")

        ocr_conf = get_ocr_confidence_for_field(bbox, ocr_results)
        llm_conf = llm_ratings.get(field_name)
        rule_passed = run_rule_checks(field_name, val, ftype)

        conf_obj = combine_confidence(
            ocr_conf=ocr_conf,
            llm_conf=llm_conf,
            rule_passed=rule_passed,
        )
        finfo["confidence"] = conf_obj

    # 3. Score table cells & calculate table_confidence
    for tbl in tables:
        rows = tbl.get("rows", [])
        all_cell_scores: list[float] = []

        for r in rows:
            cells = r.get("cells", {})
            for col_name, cdata in cells.items():
                c_bbox = cdata.get("bbox")
                c_val = cdata.get("value")

                # If cell has bbox, get overlapping OCR score
                if c_bbox:
                    # Cell bbox has: {x, y, width, height}, needs table page
                    page_num = tbl.get("bbox", {}).get("page", 1) if tbl.get("bbox") else 1
                    cell_bbox_with_page = dict(c_bbox)
                    cell_bbox_with_page["page"] = page_num
                    c_ocr_conf = get_ocr_confidence_for_field(cell_bbox_with_page, ocr_results)
                else:
                    c_ocr_conf = 0.80  # Default reasonable table cell confidence if unmapped

                # Rule check on cell value
                cell_rule_passed = bool(c_val is not None and str(c_val).strip() != "")

                # Cell confidence float (0-1)
                cell_combined = (c_ocr_conf * 0.7) + ((1.0 if cell_rule_passed else 0.0) * 0.3)
                if not cell_rule_passed:
                    cell_combined = min(cell_combined, 0.50)

                cell_score = round(float(cell_combined), 4)
                cdata["confidence"] = cell_score
                all_cell_scores.append(cell_score)

        if all_cell_scores:
            tbl["table_confidence"] = round(float(sum(all_cell_scores) / len(all_cell_scores)), 4)
        else:
            tbl["table_confidence"] = 0.0

    return extracted_fields, tables
