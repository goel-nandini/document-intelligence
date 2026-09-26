"""Access Control & Confidentiality Masking Module - Phase 7.

Provides:
- Role-based PIN verification for revealing sensitive data
- Generic and type-specific masking of sensitive PII fields
- Persistent SQLite audit logging for compliance and review tracking
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from typing import Any

import db

logger = logging.getLogger(__name__)

# ==============================================================================
# Credentials & Configuration
# ==============================================================================

# DEMO-ONLY hardcoded credential store; in production this would integrate with
# an enterprise identity provider (IdP), OAuth2/OIDC, or an encrypted secrets vault.
PIN_STORE: dict[str, str] = {
    "reviewer": "1234",
    "admin": "9999",
}


def verify_pin(role: str, entered_pin: str) -> bool:
    """Verify an entered PIN against the configured credential store for the specified role.

    Args:
        role: User role ("reviewer" or "admin").
        entered_pin: String or numeric PIN entered by the user.

    Returns:
        True if the role exists and the entered PIN matches, False otherwise.
    """
    if not role or entered_pin is None:
        return False

    role_key = str(role).strip().lower()
    expected_pin = PIN_STORE.get(role_key)

    if expected_pin is None:
        return False

    return str(entered_pin).strip() == str(expected_pin).strip()


def mask_value(value: Any, field_type: str = "text") -> str:
    """Produce a masked representation of a sensitive value for safe UI display.

    Formatting rules:
    - id_number / account_number: Show only last 4 characters, e.g. "XXXX-XXXX-1234".
    - name: Show first character + asterisks, e.g. "P*******".
    - date: Show fixed mask "XX/XX/XXXX".
    - Generic/Fallback: Keep first and last characters or pad with asterisks.

    Args:
        value: Original unmasked value.
        field_type: Declared or inferred field type.

    Returns:
        Masked string representation.
    """
    if value is None:
        return ""

    val_str = str(value).strip()
    if not val_str or val_str.lower() in ("null", "none"):
        return val_str

    ftype = str(field_type).strip().lower()

    if ftype in ("id_number", "account_number", "id"):
        # Strip spaces and hyphens to find alphanumeric characters
        clean_chars = re.sub(r"[\s-]", "", val_str)
        if len(clean_chars) <= 4:
            return "XXXX-XXXX-" + clean_chars
        last4 = clean_chars[-4:]
        return f"XXXX-XXXX-{last4}"

    elif ftype == "name":
        # Handle single or multi-word names: "John Doe" -> "J*** D***" or "J*******"
        words = val_str.split()
        if len(words) > 1:
            masked_words = [w[0] + "*" * max(len(w) - 1, 3) if len(w) > 0 else "*" for w in words]
            return " ".join(masked_words)
        else:
            return val_str[0] + "*" * max(len(val_str) - 1, 6)

    elif ftype == "date":
        return "XX/XX/XXXX"

    elif ftype in ("amount", "salary", "pay"):
        # Mask digits but preserve currency symbol if present
        currency_match = re.match(r"^([\$₹€£¥A-Z]{1,3}\s*)", val_str)
        curr = currency_match.group(1) if currency_match else ""
        return f"{curr}**,***.**"

    else:
        # Generic string masking
        if len(val_str) <= 4:
            return "*" * len(val_str)
        return val_str[:1] + "*" * (len(val_str) - 2) + val_str[-1:]


# ==============================================================================
# Audit Logging
# ==============================================================================

def log_audit_event(
    document_id: str,
    action: str,
    actor: str,
    detail: str,
    timestamp: str | None = None,
) -> None:
    """Insert an audit event into the SQLite audit_log table.

    Args:
        document_id: Identifier of the document associated with this event.
        action: Event action identifier (e.g. 'document_uploaded', 'confidential_field_revealed').
        actor: Entity performing the action (e.g. 'system', 'reviewer', 'admin').
        detail: Human-readable context and explanation of the event.
        timestamp: Optional ISO-8601 timestamp string. If omitted, uses current UTC time.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    try:
        # Ensure database and table are initialized
        db.init_db()

        with db.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO audit_log (document_id, action, actor, timestamp, detail)
                VALUES (?, ?, ?, ?, ?)
                """,
                (document_id, action, actor, timestamp, detail),
            )
            conn.commit()
            logger.info(f"Audit event logged: [{action}] by [{actor}] for doc [{document_id}]")
    except Exception as e:
        logger.error(f"Failed to record audit event: {e}", exc_info=True)


def get_audit_log(document_id: str) -> list[dict[str, Any]]:
    """Retrieve all audit log entries for a document ordered chronologically.

    Args:
        document_id: Identifier of the document.

    Returns:
        List of audit event dictionaries conforming to Phase 7 schema:
        [
            {
                "action": str,
                "actor": str,
                "timestamp": str,
                "detail": str,
            },
            ...
        ]
    """
    try:
        db.init_db()
        with db.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT action, actor, timestamp, detail
                FROM audit_log
                WHERE document_id = ?
                ORDER BY id ASC
                """,
                (document_id,),
            )
            rows = cursor.fetchall()
            return [
                {
                    "action": r["action"],
                    "actor": r["actor"],
                    "timestamp": r["timestamp"],
                    "detail": r["detail"] or "",
                }
                for r in rows
            ]
    except Exception as e:
        logger.error(f"Failed to fetch audit log for document {document_id}: {e}", exc_info=True)
        return []
