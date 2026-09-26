"""Query Chatbot Module - Phase 9 (Retrieval-Based Assistant).

Answers natural language questions about processed documents using structured data
stored in SQLite (documents table), with confidence level citations, confidentiality
masking compliance, and source document attribution. Decoupled from Streamlit session state.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import db
import extraction

logger = logging.getLogger(__name__)


def get_all_processed_documents() -> list[dict[str, Any]]:
    """Retrieve all processed documents with valid extraction results from SQLite.

    Returns:
        List of parsed document dictionaries containing full pipeline results.
    """
    db.init_db()
    documents = []
    try:
        with db.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT document_id, doc_type, upload_timestamp, full_result_json
                FROM documents
                WHERE full_result_json IS NOT NULL AND full_result_json != ''
                ORDER BY rowid DESC
                """
            )
            rows = cursor.fetchall()

        for r in rows:
            raw_json = r["full_result_json"]
            try:
                doc_dict = json.loads(raw_json)
                if isinstance(doc_dict, dict):
                    # Ensure document_id, doc_type, upload_timestamp are present
                    doc_dict.setdefault("document_id", r["document_id"])
                    doc_dict.setdefault("doc_type", r["doc_type"] or "other")
                    doc_dict.setdefault("upload_timestamp", r["upload_timestamp"])
                    documents.append(doc_dict)
            except Exception as parse_err:
                logger.warning(f"Skipping unparseable document record {r['document_id']}: {parse_err}")

    except Exception as e:
        logger.error(f"Failed to fetch processed documents: {e}", exc_info=True)

    return documents


def build_context_for_query(documents: list[dict[str, Any]], query: str = "") -> str:
    """Build a clean, readable text summary across processed documents for LLM retrieval.

    Respects confidentiality rules: if is_masked is True, displays
    '[masked — verification required]' rather than raw sensitive data.
    Omits pixel bounding boxes and raw OCR strings to keep context dense and human-readable.

    Args:
        documents: List of parsed document dictionaries.
        query: User question to tailor table summaries if relevant.

    Returns:
        Structured plain text document summary block (NOT raw JSON).
    """
    if not documents:
        return "No documents processed yet."

    summaries = []
    q_lower = query.lower() if query else ""

    for doc in documents:
        doc_id = doc.get("document_id", "unknown")
        doc_type = doc.get("doc_type", "unknown")
        upload_ts = doc.get("upload_timestamp", "")
        date_str = upload_ts[:10] if len(upload_ts) >= 10 else upload_ts

        meta = doc.get("file_metadata", {})
        fname = meta.get("original_filename") or f"{doc_type}_{doc_id[:8]}"
        lines = [f"Document [filename: {fname}, type: {doc_type}, uploaded {date_str}, ID: {doc_id}]:"]

        # 1. Extracted Fields
        fields = doc.get("extracted_fields", {})
        if fields and isinstance(fields, dict):
            for fname, finfo in fields.items():
                if not isinstance(finfo, dict):
                    continue

                is_masked = finfo.get("is_masked", False)
                if is_masked:
                    val_str = "[masked — verification required]"
                else:
                    raw_val = finfo.get("value")
                    val_str = str(raw_val) if raw_val is not None else "null"

                conf_obj = finfo.get("confidence", {})
                bucket = conf_obj.get("bucket", "unknown")
                score = conf_obj.get("combined_score")
                score_desc = f", score: {score:.2f}" if score is not None else ""
                verified_desc = ", manually verified" if finfo.get("manually_corrected") else ""

                lines.append(f"- {fname}: {val_str} (confidence: {bucket}{score_desc}{verified_desc})")

        # 2. Extracted Tables
        tables = doc.get("tables", [])
        if tables and isinstance(tables, list):
            for tbl in tables:
                tname = tbl.get("table_name", "table")
                rows = tbl.get("rows", [])
                lines.append(f"- {tname} table: {len(rows)} rows")
                for r_idx, r in enumerate(rows[:10]):
                    cells = r.get("cells", {})
                    cell_desc = []
                    for col_name, c_data in cells.items():
                        c_val = c_data.get("value") if isinstance(c_data, dict) else c_data
                        if c_val is not None:
                            cell_desc.append(f"{col_name}: {c_val}")
                    if cell_desc:
                        lines.append(f"  * row {r_idx + 1}: {', '.join(cell_desc)}")

        # 3. Validation Summary
        val = doc.get("validation", {})
        cross_checks = val.get("cross_checks", [])
        failed_checks = [c.get("check_name", "check") for c in cross_checks if not c.get("passed", True)]
        tamper_flags = val.get("tamper_flags", [])

        if failed_checks:
            lines.append(
                f"- Validation: {len(failed_checks)} cross-check(s) failed ({', '.join(failed_checks)}), "
                f"{len(tamper_flags)} tamper anomaly alert(s)"
            )
        else:
            lines.append(f"- Validation: all cross-checks passed, {len(tamper_flags)} tamper anomaly alert(s)")

        # 4. Confidentiality Status
        conf = doc.get("confidentiality", {})
        if conf.get("is_confidential", False):
            sens = conf.get("sensitivity_level", "confidential").upper()
            reason = conf.get("classification_reason", "")
            lines.append(f"- Confidentiality: CONFIDENTIAL ({sens} sensitivity — {reason})")
        else:
            lines.append("- Confidentiality: not confidential")

        # 5. Overall Review Status
        rstatus = doc.get("review_status", {})
        if isinstance(rstatus, dict):
            overall = rstatus.get("overall_status", "unknown")
            corrs = rstatus.get("corrections", [])
            lines.append(f"- Review Status: {overall} ({len(corrs)} correction(s) recorded)")

        summaries.append("\n".join(lines))

    context_str = "\n\n".join(summaries)

    # Print readable text summary to console for validation evidence
    print("\n" + "=" * 60)
    print(f"[DEBUG chatbot] GENERATED DOCUMENT CONTEXT FOR QUERY: '{query}'")
    print(f"[Type: {type(context_str).__name__}, Length: {len(context_str)} chars]")
    print("-" * 60)
    print(context_str)
    print("=" * 60 + "\n")

    return context_str


def answer_query(
    query: str,
    chat_history: list[dict[str, str]] | None = None,
    client: Any = None,
) -> dict[str, Any]:
    """Answer a user query strictly using the structured document intelligence in SQLite.

    Args:
        query: Natural language question from the user.
        chat_history: Optional list of previous chat messages: [{"role": "user"|"assistant", "content": str}].
        client: Optional Gemini API client.

    Returns:
        Dict conforming to Phase 9 specifications:
        {
            "answer": str,
            "referenced_documents": list[str], # list of document_ids
        }
    """
    documents = get_all_processed_documents()

    if not documents:
        return {
            "answer": "No documents have been processed in the system yet. Please upload a document first to start asking questions.",
            "referenced_documents": [],
        }

    # Build readable context summary
    context = build_context_for_query(documents, query=query)

    # Build history context snippet (last 4 turns)
    history_lines = []
    if chat_history:
        for turn in chat_history[-4:]:
            role = turn.get("role", "user").title()
            content = turn.get("content", "").strip()
            history_lines.append(f"{role}: {content}")
    history_block = "\n".join(history_lines) if history_lines else "None (New conversation)"

    prompt = (
        "You are an expert Document Intelligence Assistant. You answer questions strictly and accurately "
        "using ONLY the processed document summaries provided below.\n\n"
        "### STRICT INSTRUCTIONS:\n"
        "1. Answer ONLY based on the facts provided in the DOCUMENT CONTEXT below. Do NOT assume, infer, or hallucinate.\n"
        "2. If the information requested is NOT present in the provided documents, you MUST clearly state: "
        "'I don't have that information in the processed documents.'\n"
        "3. When citing a specific value (e.g. invoice total, vendor name, dates), always cite which document "
        "(by doc_type, upload date, or document ID) it came from, and mention the confidence level "
        "(e.g. 'the total amount is $45,000, which was extracted with amber/medium confidence, so you may want to verify it').\n"
        "4. If a field value is '[masked — verification required]', explain that the field is confidential and masked.\n"
        "5. Be concise, professional, and clear.\n"
        "6. At the very end of your response, on a new line, list any referenced document IDs in this format: "
        "REFERENCED_DOCS: [\"<id1>\", \"<id2>\"] (or REFERENCED_DOCS: [] if none).\n\n"
        f"### RECENT CHAT HISTORY:\n{history_block}\n\n"
        f"### PROCESSED DOCUMENT CONTEXT:\n{context}\n\n"
        f"### USER QUESTION:\n{query}"
    )

    try:
        raw_response = extraction._generate_with_gemini(prompt=prompt, client=client)
        raw_text = raw_response.strip()

        # Extract referenced document IDs if provided in the tag
        referenced_docs: list[str] = []
        ref_match = re.search(r"REFERENCED_DOCS:\s*(\[[^\]]*\])", raw_text, re.IGNORECASE)
        clean_answer = raw_text

        if ref_match:
            clean_answer = raw_text[:ref_match.start()].strip()
            json_ids = ref_match.group(1)
            try:
                parsed_ids = json.loads(json_ids)
                if isinstance(parsed_ids, list):
                    referenced_docs = [str(i) for i in parsed_ids]
            except Exception:
                pass

        # Also identify document IDs mentioned in the answer text or context
        for doc in documents:
            did = doc.get("document_id")
            if did and (did in clean_answer or did[:8] in clean_answer or did in referenced_docs):
                if did not in referenced_docs:
                    referenced_docs.append(did)

        # Fallback: if only 1 document exists and answer is not a negative fallback, cite it
        if not referenced_docs and len(documents) == 1 and "don't have that information" not in clean_answer.lower():
            referenced_docs.append(documents[0]["document_id"])

        return {
            "answer": clean_answer,
            "referenced_documents": referenced_docs,
        }

    except Exception as e:
        logger.error(f"Error answering chatbot query: {e}", exc_info=True)
        return {
            "answer": "Sorry, I couldn't process that question right now. Please try again.",
            "referenced_documents": [],
        }
