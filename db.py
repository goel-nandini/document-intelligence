"""Database and Deduplication Module - Phase 3.

Provides SQLite storage, SHA-256 exact matching, and perceptual hash similarity
search for document deduplication and pipeline result caching.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from typing import Any

import imagehash

logger = logging.getLogger(__name__)

# Default database location in the workspace root
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents.db")


def get_db_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Create and return an SQLite connection with Row factory configured."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    """Initialize the documents table and indexes if they do not exist."""
    try:
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    file_hash_sha256 TEXT NOT NULL,
                    perceptual_hash TEXT NOT NULL,
                    upload_timestamp TEXT NOT NULL,
                    full_result_json TEXT,
                    doc_type TEXT
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_file_hash ON documents (file_hash_sha256)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_perceptual_hash ON documents (perceptual_hash)"
            )
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}", exc_info=True)


# Initialize DB on import
init_db()


def save_document_record(
    document_id: str,
    file_hash: str,
    perceptual_hash: str,
    upload_timestamp: str,
    result_json: dict[str, Any] | str | None = None,
    doc_type: str | None = None,
    db_path: str = DB_PATH,
) -> bool:
    """Insert or update a document record in SQLite.

    Args:
        document_id: UUID string of the document.
        file_hash: SHA-256 hash string of raw file bytes.
        perceptual_hash: Perceptual hash string of document page(s).
        upload_timestamp: ISO-8601 upload timestamp.
        result_json: Optional dict or JSON string of pipeline results.
        doc_type: Optional document classification type (future phases).
        db_path: Path to SQLite database file.

    Returns:
        True if record saved successfully, False otherwise.
    """
    try:
        json_str: str | None = None
        if result_json is not None:
            if isinstance(result_json, str):
                json_str = result_json
            else:
                json_str = json.dumps(result_json)

        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO documents (
                    document_id, file_hash_sha256, perceptual_hash,
                    upload_timestamp, full_result_json, doc_type
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    file_hash_sha256 = excluded.file_hash_sha256,
                    perceptual_hash = excluded.perceptual_hash,
                    upload_timestamp = excluded.upload_timestamp,
                    full_result_json = coalesce(excluded.full_result_json, documents.full_result_json),
                    doc_type = coalesce(excluded.doc_type, documents.doc_type)
                """,
                (
                    document_id,
                    file_hash,
                    perceptual_hash,
                    upload_timestamp,
                    json_str,
                    doc_type,
                ),
            )
            conn.commit()
            return True
    except Exception as e:
        logger.error(f"Error saving document record {document_id}: {e}", exc_info=True)
        return False


def find_duplicate(
    file_hash: str,
    perceptual_hash: str,
    similarity_threshold: int = 5,
    db_path: str = DB_PATH,
) -> dict[str, Any] | None:
    """Search for existing duplicates via exact SHA-256 hash or perceptual hash similarity.

    Checks:
    1. Exact SHA-256 match -> match_type: 'exact_hash'
    2. Perceptual hash Hamming distance <= similarity_threshold -> match_type: 'perceptual_hash_similarity'

    Args:
        file_hash: SHA-256 hexadecimal hash string.
        perceptual_hash: Hexadecimal perceptual hash string (from imagehash).
        similarity_threshold: Maximum Hamming distance for perceptual equivalence (default: 5).
        db_path: Path to SQLite database file.

    Returns:
        Dict with duplicate details if found:
            {
                "is_duplicate": True,
                "matched_document_id": str,
                "match_type": "exact_hash" | "perceptual_hash_similarity",
                "distance": int (optional)
            }
        Or None if no duplicate is found.
    """
    try:
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()

            # 1. Exact SHA-256 hash match check
            cursor.execute(
                "SELECT document_id, upload_timestamp FROM documents WHERE file_hash_sha256 = ? LIMIT 1",
                (file_hash,),
            )
            exact_match = cursor.fetchone()
            if exact_match:
                return {
                    "is_duplicate": True,
                    "matched_document_id": exact_match["document_id"],
                    "match_type": "exact_hash",
                }

            # 2. Perceptual hash Hamming distance check
            if not perceptual_hash:
                return None

            cursor.execute(
                "SELECT document_id, perceptual_hash FROM documents WHERE perceptual_hash IS NOT NULL AND perceptual_hash != ''"
            )
            rows = cursor.fetchall()

            curr_hash = imagehash.hex_to_hash(perceptual_hash)

            best_match_id: str | None = None
            min_distance: int = 9999

            for row in rows:
                stored_p_hash_str = row["perceptual_hash"]
                try:
                    stored_hash = imagehash.hex_to_hash(stored_p_hash_str)
                    dist = int(curr_hash - stored_hash)
                    if dist <= similarity_threshold and dist < min_distance:
                        min_distance = dist
                        best_match_id = row["document_id"]
                except Exception:
                    continue

            if best_match_id is not None:
                return {
                    "is_duplicate": True,
                    "matched_document_id": best_match_id,
                    "match_type": "perceptual_hash_similarity",
                    "distance": min_distance,
                }

    except Exception as e:
        logger.error(f"Error checking document duplicates: {e}", exc_info=True)

    return None


def get_cached_result(document_id: str, db_path: str = DB_PATH) -> dict[str, Any] | None:
    """Retrieve and parse full_result_json for a given document_id.

    Args:
        document_id: Document UUID string.
        db_path: Path to SQLite database file.

    Returns:
        Parsed dictionary of cached pipeline result, or None if unavailable.
    """
    try:
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT full_result_json FROM documents WHERE document_id = ?",
                (document_id,),
            )
            row = cursor.fetchone()
            if row and row["full_result_json"]:
                return json.loads(row["full_result_json"])
    except Exception as e:
        logger.error(f"Error fetching cached result for {document_id}: {e}", exc_info=True)

    return None


def update_cached_result(
    document_id: str,
    result_json: dict[str, Any] | str,
    db_path: str = DB_PATH,
) -> bool:
    """Update full_result_json for an existing document record.

    Args:
        document_id: Document UUID string.
        result_json: Dict or JSON string of pipeline results.
        db_path: Path to SQLite database file.

    Returns:
        True if updated, False otherwise.
    """
    try:
        json_str = result_json if isinstance(result_json, str) else json.dumps(result_json)
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE documents SET full_result_json = ? WHERE document_id = ?",
                (json_str, document_id),
            )
            conn.commit()
            return cursor.rowcount > 0
    except Exception as e:
        logger.error(f"Error updating cached result for {document_id}: {e}", exc_info=True)
        return False
