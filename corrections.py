"""Correction Memory & Human-in-the-Loop Module - Phase 8.

Provides:
- SQLite persistence of reviewer corrections
- Audit logging of field correction events
- Few-shot correction memory retrieval for LLM prompt augmentation
- Review accuracy and correction rate metrics calculation
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import access_control
import db

logger = logging.getLogger(__name__)


def save_correction(
    document_id: str,
    doc_type: str,
    field_name: str,
    original_value: Any,
    corrected_value: Any,
    corrected_by: str = "Reviewer",
    ocr_context_snippet: str = "",
) -> dict[str, str]:
    """Persist a human reviewer correction, log an audit event, and return the schema-compliant dict.

    Args:
        document_id: Unique document identifier.
        doc_type: Document classification type (e.g. 'invoice', 'payslip').
        field_name: The extracted field being corrected.
        original_value: System-extracted or previous value.
        corrected_value: New reviewer-provided value.
        corrected_by: Name or role of the reviewer (default: 'Reviewer').
        ocr_context_snippet: Raw OCR text snippet context around this field.

    Returns:
        Dict conforming to Phase 8 review_status.corrections item schema:
        {
            "field_name": str,
            "original_value": str,
            "corrected_value": str,
            "corrected_by": str,
            "corrected_at": str (ISO-8601),
        }
    """
    db.init_db()
    corrected_at = datetime.now(timezone.utc).isoformat()
    orig_str = str(original_value) if original_value is not None else ""
    corr_str = str(corrected_value).strip() if corrected_value is not None else ""
    actor_str = str(corrected_by).strip() if corrected_by else "Reviewer"
    snippet_str = str(ocr_context_snippet).strip() if ocr_context_snippet else ""

    try:
        with db.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO corrections (
                    document_id, doc_type, field_name, original_value,
                    corrected_value, corrected_by, corrected_at, ocr_context_snippet
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    doc_type,
                    field_name,
                    orig_str,
                    corr_str,
                    actor_str,
                    corrected_at,
                    snippet_str,
                ),
            )
            conn.commit()

        # Log audit event
        access_control.log_audit_event(
            document_id=document_id,
            action="field_corrected",
            actor=actor_str,
            detail=f"Field '{field_name}' corrected from '{orig_str}' to '{corr_str}'",
            timestamp=corrected_at,
        )

        logger.info(
            f"Correction recorded for doc [{document_id}], field [{field_name}]: '{orig_str}' -> '{corr_str}'"
        )
    except Exception as e:
        logger.error(f"Failed to save correction to database: {e}", exc_info=True)
        raise

    return {
        "field_name": field_name,
        "original_value": orig_str,
        "corrected_value": corr_str,
        "corrected_by": actor_str,
        "corrected_at": corrected_at,
    }


def get_correction_examples(
    doc_type: str,
    field_name: str,
    limit: int = 3,
) -> list[dict[str, str]]:
    """Fetch the most recent corrections for a specific doc_type + field_name.

    Args:
        doc_type: Document classification type.
        field_name: Target field name.
        limit: Maximum number of historical examples to return.

    Returns:
        List of dicts with keys: ocr_context_snippet, original_value, corrected_value.
    """
    db.init_db()
    try:
        with db.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT ocr_context_snippet, original_value, corrected_value
                FROM corrections
                WHERE doc_type = ? AND field_name = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (doc_type, field_name, limit),
            )
            rows = cursor.fetchall()
            return [
                {
                    "ocr_context_snippet": r["ocr_context_snippet"] or "",
                    "original_value": r["original_value"] or "",
                    "corrected_value": r["corrected_value"] or "",
                }
                for r in rows
            ]
    except Exception as e:
        logger.error(f"Failed to fetch correction examples: {e}", exc_info=True)
        return []


def build_fewshot_examples_block(doc_type: str, limit: int = 5) -> str:
    """Fetch recent corrections across all fields for doc_type and format into prompt memory text.

    Args:
        doc_type: Document classification type.
        limit: Maximum total recent corrections to include.

    Returns:
        Formatted multi-line fewshot guidance string, or empty string if no corrections exist.
    """
    db.init_db()
    try:
        with db.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT field_name, original_value, corrected_value, ocr_context_snippet
                FROM corrections
                WHERE doc_type = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (doc_type, limit),
            )
            rows = cursor.fetchall()

        if not rows:
            return ""

        lines = []
        for r in rows:
            fname = r["field_name"]
            orig_val = r["original_value"] or ""
            corr_val = r["corrected_value"] or ""
            snippet = r["ocr_context_snippet"] or ""

            # Clean snippet for prompt injection safety
            clean_snippet = snippet.replace("\n", " ").strip()
            if len(clean_snippet) > 80:
                clean_snippet = clean_snippet[:77] + "..."

            lines.append(
                f"- Note: for documents like this, past corrections show: when text similar to '{clean_snippet}' "
                f"appeared, the correct value for '{fname}' was '{corr_val}' "
                f"(previously extracted incorrectly as '{orig_val}')."
            )

        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Failed to build fewshot examples block: {e}", exc_info=True)
        return ""


def get_accuracy_stats(doc_type: str | None = None) -> dict[str, Any]:
    """Compute correction metrics and time-series trend for judge evaluation.

    Args:
        doc_type: Optional filter by document type.

    Returns:
        Dict containing:
        - total_documents_processed: int
        - total_corrections_made: int
        - correction_rate_trend: list of {"date": str, "correction_count": int}
    """
    db.init_db()
    try:
        with db.get_db_connection() as conn:
            cursor = conn.cursor()

            # 1. Total documents processed
            if doc_type:
                cursor.execute("SELECT COUNT(*) FROM documents WHERE doc_type = ?", (doc_type,))
            else:
                cursor.execute("SELECT COUNT(*) FROM documents")
            total_docs = cursor.fetchone()[0]

            # 2. Total corrections made
            if doc_type:
                cursor.execute("SELECT COUNT(*) FROM corrections WHERE doc_type = ?", (doc_type,))
            else:
                cursor.execute("SELECT COUNT(*) FROM corrections")
            total_corrections = cursor.fetchone()[0]

            # 3. Corrections trend grouped by date (YYYY-MM-DD)
            if doc_type:
                cursor.execute(
                    """
                    SELECT SUBSTR(corrected_at, 1, 10) AS dt, COUNT(*) AS cnt
                    FROM corrections
                    WHERE doc_type = ?
                    GROUP BY dt
                    ORDER BY dt ASC
                    """,
                    (doc_type,),
                )
            else:
                cursor.execute(
                    """
                    SELECT SUBSTR(corrected_at, 1, 10) AS dt, COUNT(*) AS cnt
                    FROM corrections
                    GROUP BY dt
                    ORDER BY dt ASC
                    """
                )
            trend_rows = cursor.fetchall()

            trend = [{"date": r["dt"], "correction_count": r["cnt"]} for r in trend_rows]

            return {
                "total_documents_processed": total_docs,
                "total_corrections_made": total_corrections,
                "correction_rate_trend": trend,
            }
    except Exception as e:
        logger.error(f"Failed to get accuracy stats: {e}", exc_info=True)
        return {
            "total_documents_processed": 0,
            "total_corrections_made": 0,
            "correction_rate_trend": [],
        }
