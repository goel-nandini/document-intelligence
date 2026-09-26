"""Authenticity Pre-Check and Tamper Detection Module - Phase 3.

Provides lightweight pre-checks for document authenticity:
- Perceptual image hashing via imagehash (pHash)
- Metadata timeline consistency and suspicious editing software detection
- Error Level Analysis (ELA) for recompression artifacts and localized manipulation
- Flag synthesis conforming to the Phase 3 validation schema.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timezone
from typing import Any

import imagehash
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ==============================================================================
# Named Constants & Tunable Thresholds
# ==============================================================================
# Threshold in days between document creation and modification before flagging
METADATA_TIME_GAP_THRESHOLD_DAYS: float = 1.0

# Known document editing, photo manipulation, or synthetic rendering tools
SUSPICIOUS_SOFTWARE_KEYWORDS: list[str] = [
    "photoshop",
    "gimp",
    "paint",
    "paint.net",
    "canva",
    "illustrator",
    "inkscape",
    "coreldraw",
    "acrobat distiller",
    "ilovepdf",
    "sejda",
    "pdfescape",
    "nitro",
    "pdf editor",
    "pixlr",
]

# Error Level Analysis (ELA) parameters
# Note: Error Level Analysis is a lightweight visual heuristic based on compression
# differential, not a definitive forensic proof. It highlights localized compression
# disparities that warrant manual human review or specialized forensic inspection.
ELA_JPEG_QUALITY: int = 90
ELA_ANOMALY_DIFF_THRESHOLD: float = 25.0
ELA_ANOMALY_PERCENTAGE_THRESHOLD: float = 1.50


def compute_perceptual_hash(image: np.ndarray) -> str:
    """Compute perceptual hash (pHash) of an image.

    Converts the image to a PIL Image and runs imagehash.phash.
    pHash captures the low-frequency discrete cosine transform structure,
    making it resilient to minor recompression, scaling, and lighting variations.

    Args:
        image: RGB image as numpy array.

    Returns:
        Hexadecimal perceptual hash string (e.g. '8f3c2e1a...').
    """
    try:
        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_img = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        phash = imagehash.phash(pil_img)
        return str(phash)
    except Exception as e:
        logger.warning(f"Error computing perceptual hash: {e}", exc_info=True)
        return "0000000000000000"


def _parse_iso_date(dt_val: str | None) -> datetime | None:
    """Parse ISO datetime string safely into a timezone-aware UTC datetime."""
    if not dt_val:
        return None
    try:
        # Normalize trailing 'Z' if present
        cleaned = str(dt_val).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def check_metadata_consistency(file_metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Inspect document metadata for temporal inconsistencies and suspicious software.

    Flags:
    - Time Gap: Modification date significantly later than creation date.
    - Suspicious Software: Document produced by graphic design/image editing software.

    Args:
        file_metadata: Dict matching file_metadata schema.

    Returns:
        List of tamper flag dicts matching schema:
            flag_type: 'metadata_mismatch'
            affected_field: None
            evidence: str
            severity: 'low' | 'medium' | 'high'
    """
    flags: list[dict[str, Any]] = []

    try:
        creation_str = file_metadata.get("creation_date_from_metadata")
        modification_str = file_metadata.get("modification_date_from_metadata")
        software_used = file_metadata.get("software_used")

        # 1. Timeline check between creation and modification dates
        c_dt = _parse_iso_date(creation_str)
        m_dt = _parse_iso_date(modification_str)

        if c_dt and m_dt:
            gap_seconds = (m_dt - c_dt).total_seconds()
            gap_days = gap_seconds / 86400.0

            if gap_days > METADATA_TIME_GAP_THRESHOLD_DAYS:
                severity = "high" if gap_days > 14.0 else "medium"
                flags.append({
                    "flag_type": "metadata_mismatch",
                    "affected_field": None,
                    "evidence": (
                        f"Document modification date ({modification_str[:19]}) is {gap_days:.1f} days "
                        f"after creation date ({creation_str[:19]}), suggesting retrospective edits."
                    ),
                    "severity": severity,
                })
            elif gap_days < -0.01:
                # Modification prior to creation is an anomaly
                flags.append({
                    "flag_type": "metadata_mismatch",
                    "affected_field": None,
                    "evidence": (
                        f"Modification timestamp ({modification_str[:19]}) precedes creation timestamp "
                        f"({creation_str[:19]}), indicating clock skew or metadata tampering."
                    ),
                    "severity": "high",
                })

        # 2. Suspicious software check
        if software_used:
            software_lower = str(software_used).lower()
            matched_tool = next(
                (tool for tool in SUSPICIOUS_SOFTWARE_KEYWORDS if tool in software_lower),
                None,
            )
            if matched_tool:
                flags.append({
                    "flag_type": "metadata_mismatch",
                    "affected_field": None,
                    "evidence": (
                        f"Producer metadata identifies photo-editing/authoring software: '{software_used}' "
                        f"(matched keyword '{matched_tool}'). Authentic statements/invoices typically "
                        f"originate from ERP, billing, or PDF library generators."
                    ),
                    "severity": "high",
                })

    except Exception as e:
        logger.warning(f"Error during metadata consistency check: {e}", exc_info=True)

    return flags


def error_level_analysis(image: np.ndarray) -> dict[str, Any]:
    """Execute Error Level Analysis (ELA) to detect localized compression differences.

    ELA resaves the image at a known JPEG quality (e.g. 90) and measures the
    absolute pixel discrepancy. In genuine unaltered images, recompression error
    is relatively uniform across surfaces of equal frequency. Localized inserts,
    spliced text, or pasted numerical amounts often display distinct error levels.

    NOTE: This is a lightweight heuristic signal for review triage, not a definitive
    forensic guarantee.

    Args:
        image: RGB image as numpy array.

    Returns:
        Dict containing:
            - ela_score: float (mean difference intensity across all pixels)
            - suspicious_regions_detected: bool (True if anomalous high-error pixel ratio exceeds threshold)
    """
    try:
        if not isinstance(image, np.ndarray) or image.size == 0:
            return {"ela_score": 0.0, "suspicious_regions_detected": False}

        pil_orig = Image.fromarray(image).convert("RGB")

        # Resave in-memory at standard quality
        buffer = io.BytesIO()
        pil_orig.save(buffer, format="JPEG", quality=ELA_JPEG_QUALITY)
        buffer.seek(0)
        resaved_img = Image.open(buffer).convert("RGB")

        orig_arr = np.array(pil_orig, dtype=np.float32)
        resaved_arr = np.array(resaved_img, dtype=np.float32)

        # Absolute difference per channel
        abs_diff = np.abs(orig_arr - resaved_arr)
        # Average difference across channels
        channel_mean_diff = np.mean(abs_diff, axis=2)

        ela_score = round(float(np.mean(channel_mean_diff)), 2)

        # Compute ratio of pixels with abnormally large recompression delta
        high_diff_mask = channel_mean_diff > ELA_ANOMALY_DIFF_THRESHOLD
        anomalous_pixel_percentage = float(np.mean(high_diff_mask) * 100.0)

        # Suspicious if anomalous percentage exceeds threshold
        is_suspicious = bool(anomalous_pixel_percentage > ELA_ANOMALY_PERCENTAGE_THRESHOLD)

        return {
            "ela_score": ela_score,
            "suspicious_regions_detected": is_suspicious,
            "anomalous_pixel_percentage": round(anomalous_pixel_percentage, 2),
        }

    except Exception as e:
        logger.warning(f"Error executing Error Level Analysis: {e}", exc_info=True)
        return {"ela_score": 0.0, "suspicious_regions_detected": False}


def run_authenticity_precheck(
    image: np.ndarray,
    file_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """Orchestrate authenticity pre-checks and return structured tamper flags.

    Executes:
    1. Metadata consistency inspection (dates + software keywords).
    2. Error Level Analysis on the primary page.

    Args:
        image: RGB page image as numpy array.
        file_metadata: Dict matching file_metadata schema.

    Returns:
        List of tamper flag dicts conforming to Phase 3 schema:
        [
          {
            "flag_type": "metadata_mismatch | recompression_artifact",
            "affected_field": None,
            "evidence": str,
            "severity": "low | medium | high"
          }
        ]
    """
    flags: list[dict[str, Any]] = []

    # 1. Metadata Consistency
    try:
        meta_flags = check_metadata_consistency(file_metadata)
        flags.extend(meta_flags)
    except Exception as e:
        logger.warning(f"Metadata consistency pre-check failed: {e}", exc_info=True)

    # 2. Error Level Analysis
    try:
        ela_res = error_level_analysis(image)
        if ela_res.get("suspicious_regions_detected", False):
            flags.append({
                "flag_type": "recompression_artifact",
                "affected_field": None,
                "evidence": (
                    f"Error Level Analysis detected localized compression disparity "
                    f"(ELA Score: {ela_res['ela_score']}, {ela_res.get('anomalous_pixel_percentage', 0)}% "
                    f"high-variance pixels), suggesting potential localized re-saving or spliced content."
                ),
                "severity": "medium",
            })
    except Exception as e:
        logger.warning(f"ELA pre-check failed: {e}", exc_info=True)

    return flags
