"""Validation Layer Module - Phase 6.

Provides:
- Business logic cross-checks (invoice line items sum, payslip deductions vs net pay, date chronology)
- Font-consistency tamper detection (character height and stroke width variance analysis)
- Document confidentiality and PII classification
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ==============================================================================
# Named Constants & Configuration
# ==============================================================================

# Font consistency deviation threshold (in standard deviations)
FONT_DEVIATION_THRESHOLD: float = 2.0

# Document classification to default sensitivity level mapping
CONFIDENTIAL_DOC_TYPES: dict[str, str] = {
    "id_proof": "high",
    "payslip": "high",
    "bank_statement": "high",
    "invoice": "low",
    "other": "low",
}

# PII field patterns: mapping field_type and field_name keywords that count as PII
PII_FIELD_PATTERNS: dict[str, list[str]] = {
    "field_types": [
        "id_number",
    ],
    "field_names": [
        "id_number",
        "passport",
        "ssn",
        "social_security",
        "license",
        "tax_id",
        "pan",
        "aadhaar",
        "voter",
        "account_number",
        "account_no",
        "iban",
        "credit_card",
        "card_number",
        "dob",
        "date_of_birth",
        "birth",
        "salary",
        "net_pay",
        "gross_pay",
        "basic_salary",
        "routing_number",
    ],
}


# ==============================================================================
# Cross-Checks
# ==============================================================================

def _parse_amount(value: Any) -> float | None:
    """Helper to extract a clean float from a currency or numeric string."""
    if value is None:
        return None
    val_str = str(value).strip()
    if not val_str or val_str.lower() in ("null", "none"):
        return None
    # Strip currency symbols and letters, keeping digits, commas, and dots
    clean_str = re.sub(r"[^\d.-]", "", val_str)
    try:
        return float(clean_str)
    except ValueError:
        return None


def _parse_date(date_str: Any) -> datetime | None:
    """Helper to parse a date string using standard formats."""
    if date_str is None:
        return None
    s = str(date_str).strip()
    if not s or s.lower() in ("null", "none"):
        return None
    cleaned = re.sub(r"[,\.]", "", s)
    formats = [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d",
        "%d %b %Y", "%d %B %Y", "%b %d %Y", "%B %d %Y",
        "%Y%m%d", "%d.%m.%Y", "%Y.%m.%d",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return None


def check_line_items_sum(
    extracted_fields: dict[str, Any],
    tables: list[dict[str, Any]],
    tolerance: float = 1.0,
) -> dict[str, Any]:
    """For invoices: sum the 'amount' column of the 'line_items' table and compare against total_amount.

    Args:
        extracted_fields: Phase 4/5 extracted fields dictionary.
        tables: Phase 4/5 extracted tables list.
        tolerance: Allowed difference for rounding (default ±1.0).

    Returns:
        Cross-check result dictionary conforming to Phase 6 schema.
    """
    check_name = "line_items_sum_vs_total"
    fields_compared = ["line_items.amount", "total_amount"]
    expected_relationship = f"Sum of line item amounts must equal total_amount (tolerance ±{tolerance:.2f})"

    # Locate line_items table
    line_items_tbl = next((t for t in tables if t.get("table_name") == "line_items"), None)
    total_amount_field = extracted_fields.get("total_amount")

    if not line_items_tbl or not total_amount_field:
        return {
            "check_name": check_name,
            "fields_compared": fields_compared,
            "expected_relationship": expected_relationship,
            "passed": True,
            "discrepancy_detail": "skipped — required fields not present",
        }

    total_amount_val = _parse_amount(total_amount_field.get("value"))
    if total_amount_val is None:
        return {
            "check_name": check_name,
            "fields_compared": fields_compared,
            "expected_relationship": expected_relationship,
            "passed": True,
            "discrepancy_detail": "skipped — required fields not present",
        }

    rows = line_items_tbl.get("rows", [])
    if not rows:
        return {
            "check_name": check_name,
            "fields_compared": fields_compared,
            "expected_relationship": expected_relationship,
            "passed": True,
            "discrepancy_detail": "skipped — required fields not present",
        }

    parsed_amounts: list[float] = []
    for r in rows:
        cells = r.get("cells", {})
        amt_cell = cells.get("amount") or cells.get("total")
        if amt_cell:
            c_val = amt_cell.get("value") if isinstance(amt_cell, dict) else amt_cell
            amt = _parse_amount(c_val)
            if amt is not None:
                parsed_amounts.append(amt)

    if not parsed_amounts:
        return {
            "check_name": check_name,
            "fields_compared": fields_compared,
            "expected_relationship": expected_relationship,
            "passed": True,
            "discrepancy_detail": "skipped — required fields not present",
        }

    sum_items = round(sum(parsed_amounts), 2)
    diff = round(abs(sum_items - total_amount_val), 2)

    if diff <= tolerance:
        return {
            "check_name": check_name,
            "fields_compared": fields_compared,
            "expected_relationship": expected_relationship,
            "passed": True,
            "discrepancy_detail": None,
        }
    else:
        return {
            "check_name": check_name,
            "fields_compared": fields_compared,
            "expected_relationship": expected_relationship,
            "passed": False,
            "discrepancy_detail": (
                f"Sum of line items ({sum_items:.2f}) does not match total_amount ({total_amount_val:.2f}). "
                f"Discrepancy: {diff:.2f} (exceeds tolerance of {tolerance:.2f})."
            ),
        }


def check_deductions_vs_net_pay(
    extracted_fields: dict[str, Any],
    tables: list[dict[str, Any]],
    tolerance: float = 1.0,
) -> dict[str, Any]:
    """For payslips: verify gross_pay - sum(deductions.amount) is reasonably close to net_pay.

    Args:
        extracted_fields: Phase 4/5 extracted fields dictionary.
        tables: Phase 4/5 extracted tables list.
        tolerance: Allowed difference for rounding (default ±1.0).

    Returns:
        Cross-check result dictionary conforming to Phase 6 schema.
    """
    check_name = "deductions_vs_net_pay"
    fields_compared = ["gross_pay", "deductions.amount", "net_pay"]
    expected_relationship = f"gross_pay minus sum of deductions must equal net_pay (tolerance ±{tolerance:.2f})"

    gross_field = extracted_fields.get("gross_pay")
    net_field = extracted_fields.get("net_pay")

    if not gross_field or not net_field:
        return {
            "check_name": check_name,
            "fields_compared": fields_compared,
            "expected_relationship": expected_relationship,
            "passed": True,
            "discrepancy_detail": "skipped — required fields not present",
        }

    gross_val = _parse_amount(gross_field.get("value"))
    net_val = _parse_amount(net_field.get("value"))

    if gross_val is None or net_val is None:
        return {
            "check_name": check_name,
            "fields_compared": fields_compared,
            "expected_relationship": expected_relationship,
            "passed": True,
            "discrepancy_detail": "skipped — required fields not present",
        }

    # Sum deductions from table or direct field
    deductions_sum = 0.0
    ded_table = next((t for t in tables if "deduction" in t.get("table_name", "").lower()), None)
    if ded_table:
        for r in ded_table.get("rows", []):
            cells = r.get("cells", {})
            amt_cell = cells.get("amount") or cells.get("deduction_amount")
            if amt_cell:
                c_val = amt_cell.get("value") if isinstance(amt_cell, dict) else amt_cell
                amt = _parse_amount(c_val)
                if amt is not None:
                    deductions_sum += amt
    elif "total_deductions" in extracted_fields:
        td_val = _parse_amount(extracted_fields["total_deductions"].get("value"))
        if td_val is not None:
            deductions_sum = td_val

    expected_net = round(gross_val - deductions_sum, 2)
    diff = round(abs(expected_net - net_val), 2)

    if diff <= tolerance:
        return {
            "check_name": check_name,
            "fields_compared": fields_compared,
            "expected_relationship": expected_relationship,
            "passed": True,
            "discrepancy_detail": None,
        }
    else:
        return {
            "check_name": check_name,
            "fields_compared": fields_compared,
            "expected_relationship": expected_relationship,
            "passed": False,
            "discrepancy_detail": (
                f"Gross pay ({gross_val:.2f}) minus deductions ({deductions_sum:.2f}) equals {expected_net:.2f}, "
                f"which does not match declared net_pay ({net_val:.2f}). Discrepancy: {diff:.2f}."
            ),
        }


def check_date_logic(extracted_fields: dict[str, Any]) -> list[dict[str, Any]]:
    """For any document with date pairs, verify that the subsequent date is on or after the prior date.

    Args:
        extracted_fields: Phase 4/5 extracted fields dictionary.

    Returns:
        List of cross-check result dictionaries.
    """
    date_pairs = [
        ("invoice_date", "due_date"),
        ("issue_date", "expiry_date"),
        ("statement_period_start", "statement_period_end"),
        ("start_date", "end_date"),
    ]

    results: list[dict[str, Any]] = []

    for start_key, end_key in date_pairs:
        if start_key in extracted_fields and end_key in extracted_fields:
            start_raw = extracted_fields[start_key].get("value")
            end_raw = extracted_fields[end_key].get("value")

            start_dt = _parse_date(start_raw)
            end_dt = _parse_date(end_raw)

            check_name = f"{start_key}_vs_{end_key}_chronology"
            fields_compared = [start_key, end_key]
            expected_relationship = f"{end_key} must be on or after {start_key}"

            if start_dt is None or end_dt is None:
                results.append({
                    "check_name": check_name,
                    "fields_compared": fields_compared,
                    "expected_relationship": expected_relationship,
                    "passed": True,
                    "discrepancy_detail": "skipped — required fields not present or unparseable",
                })
                continue

            if end_dt >= start_dt:
                results.append({
                    "check_name": check_name,
                    "fields_compared": fields_compared,
                    "expected_relationship": expected_relationship,
                    "passed": True,
                    "discrepancy_detail": None,
                })
            else:
                results.append({
                    "check_name": check_name,
                    "fields_compared": fields_compared,
                    "expected_relationship": expected_relationship,
                    "passed": False,
                    "discrepancy_detail": (
                        f"Chronological inconsistency: {end_key} ({end_dt.strftime('%Y-%m-%d')}) "
                        f"precedes {start_key} ({start_dt.strftime('%Y-%m-%d')})."
                    ),
                })

    return results


def run_all_cross_checks(
    doc_type: str,
    extracted_fields: dict[str, Any],
    tables: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Execute all relevant business logic cross-checks based on doc_type.

    Checks that do not apply to the doc_type are omitted.

    Args:
        doc_type: Document classification category.
        extracted_fields: Phase 4/5 extracted fields dictionary.
        tables: Phase 4/5 extracted tables list.

    Returns:
        List of cross_check result dictionaries.
    """
    checks: list[dict[str, Any]] = []

    if doc_type == "invoice":
        checks.append(check_line_items_sum(extracted_fields, tables))
        checks.extend(check_date_logic(extracted_fields))
    elif doc_type == "payslip":
        checks.append(check_deductions_vs_net_pay(extracted_fields, tables))
        checks.extend(check_date_logic(extracted_fields))
    elif doc_type in ("id_proof", "bank_statement", "other"):
        checks.extend(check_date_logic(extracted_fields))

    return checks


# ==============================================================================
# Font-Consistency Tamper Check
# ==============================================================================

def _estimate_stroke_width(gray_roi: np.ndarray) -> float:
    """Estimate average character stroke width in a grayscale ROI using distance transform."""
    if gray_roi.size == 0:
        return 0.0

    # Binarize with Otsu inversion (text = white, background = black)
    _, thresh = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if cv2.countNonZero(thresh) == 0:
        return 0.0

    # Distance transform computes distance to nearest zero pixel for each white pixel
    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    # Foreground pixels distance values
    foreground_dist = dist[thresh > 0]
    if len(foreground_dist) == 0:
        return 0.0

    # Stroke width is approximately 2x the distance transform peak/mean
    return float(np.mean(foreground_dist) * 2.0)


def analyze_font_consistency(
    ocr_results: dict[str, Any],
    extracted_fields: dict[str, Any],
    corrected_image: np.ndarray,
) -> list[dict[str, Any]]:
    """Analyze character height and stroke width variance across extracted fields.

    Flags any field that deviates significantly (> FONT_DEVIATION_THRESHOLD standard deviations)
    from document-wide typography baselines.

    NOTE: This is a heuristic signal, not definitive proof of tampering — false positives
    are possible on genuinely multi-font documents (e.g. a stamped date on a printed form).

    Args:
        ocr_results: Complete Phase 2 OCR dictionary.
        extracted_fields: Phase 4/5 extracted fields dictionary.
        corrected_image: OpenCV BGR/Grayscale image of the first page.

    Returns:
        List of tamper_flag dictionaries for any detected font inconsistencies.
    """
    flags: list[dict[str, Any]] = []

    if corrected_image is None or not isinstance(corrected_image, np.ndarray):
        logger.warning("Corrected image is unavailable for font consistency analysis.")
        return flags

    img_h, img_w = corrected_image.shape[:2]
    gray = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY) if len(corrected_image.shape) == 3 else corrected_image

    # 1. Compute Document-Wide Baseline across all recognized OCR words
    pages = ocr_results.get("pages", [])
    doc_words: list[dict[str, Any]] = []
    for p in pages:
        doc_words.extend(p.get("words", []))

    # Collect valid word bounding box heights
    word_heights = [
        float(w["bbox"]["height"]) for w in doc_words
        if w.get("bbox") and w["bbox"].get("height", 0) >= 6
    ]

    if len(word_heights) < 5:
        logger.info("Insufficient words (<5) to establish reliable font baseline. Skipping font check.")
        return flags

    doc_mean_height = float(np.mean(word_heights))
    doc_std_height = float(np.std(word_heights))
    # Protect against zero/tiny std dev
    doc_std_height = max(doc_std_height, 2.0)

    # Compute baseline stroke width from a sample of representative words
    sample_stroke_widths: list[float] = []
    for w in doc_words[:50]:
        wb = w.get("bbox")
        if wb:
            wx, wy = max(0, wb.get("x", 0)), max(0, wb.get("y", 0))
            ww, wh = wb.get("width", 0), wb.get("height", 0)
            if ww >= 6 and wh >= 6 and wx + ww <= img_w and wy + wh <= img_h:
                roi = gray[wy : wy + wh, wx : wx + ww]
                sw = _estimate_stroke_width(roi)
                if sw > 0.5:
                    sample_stroke_widths.append(sw)

    doc_mean_sw = float(np.mean(sample_stroke_widths)) if sample_stroke_widths else 2.5
    doc_std_sw = max(float(np.std(sample_stroke_widths)), 0.8) if sample_stroke_widths else 1.0

    # 2. Evaluate Each Extracted Field
    for fname, finfo in extracted_fields.items():
        try:
            bbox = finfo.get("bbox")
            if not bbox or not isinstance(bbox, dict):
                continue

            bx = max(0, int(bbox.get("x", 0)))
            by = max(0, int(bbox.get("y", 0)))
            bw = int(bbox.get("width", 0))
            bh = int(bbox.get("height", 0))

            # Bounds check
            if bw < 8 or bh < 8 or bx >= img_w or by >= img_h:
                continue

            crop_w = min(bw, img_w - bx)
            crop_h = min(bh, img_h - by)
            field_roi = gray[by : by + crop_h, bx : bx + crop_w]

            if field_roi.size == 0:
                continue

            # (a) Field character height analysis
            field_words = [
                w for w in doc_words
                if w.get("bbox")
                and bx <= w["bbox"].get("x", 0) <= bx + bw
                and by <= w["bbox"].get("y", 0) <= by + bh
            ]

            if field_words:
                field_mean_height = float(np.mean([w["bbox"]["height"] for w in field_words]))
            else:
                field_mean_height = float(bh)

            height_z = abs(field_mean_height - doc_mean_height) / doc_std_height

            # (b) Field stroke width analysis
            field_sw = _estimate_stroke_width(field_roi)
            sw_z = abs(field_sw - doc_mean_sw) / doc_std_sw if field_sw > 0.5 else 0.0

            # (c) Check against threshold
            max_z = max(height_z, sw_z)
            if max_z > FONT_DEVIATION_THRESHOLD:
                # Determine severity
                if max_z >= 3.5:
                    sev = "high"
                elif max_z >= 2.5:
                    sev = "medium"
                else:
                    sev = "low"

                pct_diff = abs(field_mean_height - doc_mean_height) / doc_mean_height * 100.0

                if height_z >= sw_z:
                    comparison_str = "larger" if field_mean_height > doc_mean_height else "smaller"
                    evidence = (
                        f"Character height ({field_mean_height:.1f}px) is {pct_diff:.0f}% {comparison_str} "
                        f"than document average ({doc_mean_height:.1f}px, z-score: {height_z:.2f} std dev > {FONT_DEVIATION_THRESHOLD:.1f} std dev), "
                        f"suggesting this field may have been modified or inserted with an incongruent font."
                    )
                else:
                    comparison_str = "heavier" if field_sw > doc_mean_sw else "thinner"
                    evidence = (
                        f"Character stroke width ({field_sw:.1f}px) is {comparison_str} than document average "
                        f"({doc_mean_sw:.1f}px, z-score: {sw_z:.2f} std dev > {FONT_DEVIATION_THRESHOLD:.1f} std dev), "
                        f"suggesting a font weight anomaly."
                    )

                flags.append({
                    "flag_type": "font_inconsistency",
                    "affected_field": fname,
                    "evidence": evidence,
                    "severity": sev,
                })

        except Exception as e:
            logger.warning(f"Font consistency analysis skipped for field '{fname}': {e}")
            continue

    return flags


# ==============================================================================
# Confidentiality Classification
# ==============================================================================

def classify_confidentiality(
    doc_type: str,
    extracted_fields: dict[str, Any],
) -> dict[str, Any]:
    """Classify document confidentiality and identify sensitive PII fields requiring masking.

    Args:
        doc_type: Document classification category.
        extracted_fields: Phase 4/5 extracted fields dictionary.

    Returns:
        Confidentiality dictionary conforming to Phase 6 schema contract.
    """
    matched_pii_fields: list[str] = []

    types_to_match = PII_FIELD_PATTERNS.get("field_types", ["id_number"])
    names_to_match = PII_FIELD_PATTERNS.get("field_names", [])

    # Check extracted fields against PII patterns
    for fname, finfo in extracted_fields.items():
        fname_lower = fname.lower()
        ftype_lower = str(finfo.get("field_type", "")).lower()

        # Check field type match
        if ftype_lower in types_to_match:
            matched_pii_fields.append(fname)
            continue

        # Check field name keyword match
        if any(pat in fname_lower for pat in names_to_match):
            matched_pii_fields.append(fname)

    # Deduplicate while preserving order
    matched_pii_fields = list(dict.fromkeys(matched_pii_fields))

    doc_default_sensitivity = CONFIDENTIAL_DOC_TYPES.get(doc_type, "low")
    is_high_sens_doc = (doc_default_sensitivity == "high")
    has_two_or_more_pii = (len(matched_pii_fields) >= 2)

    # Rule: If 2 or more PII-pattern fields found OR doc_type is in CONFIDENTIAL_DOC_TYPES with sensitivity "high"
    is_confidential = has_two_or_more_pii or is_high_sens_doc

    # Sensitivity level:
    # "high" if is_confidential and doc_type is in high-sensitivity list
    # "low" if is_confidential but doc_type is borderline
    # "none" if not confidential at all
    if not is_confidential:
        sensitivity_level = "none"
    elif is_high_sens_doc or len(matched_pii_fields) >= 3:
        sensitivity_level = "high"
    else:
        sensitivity_level = "low"

    # Build human-readable classification reason
    if matched_pii_fields:
        readable_fields = ", ".join(f"'{f.replace('_', ' ').title()}'" for f in matched_pii_fields)
        if is_high_sens_doc:
            reason = (
                f"Document type '{doc_type.replace('_', ' ').title()}' classified as high sensitivity; "
                f"contains protected PII fields ({readable_fields})."
            )
        else:
            reason = (
                f"PII pattern detected: document contains sensitive personal/financial fields ({readable_fields})."
            )
    elif is_high_sens_doc:
        reason = (
            f"Document classified as high sensitivity based on document category '{doc_type.replace('_', ' ').title()}'."
        )
    else:
        reason = "Standard commercial document; no sensitive PII patterns identified."

    return {
        "is_confidential": bool(is_confidential),
        "classification_reason": reason,
        "sensitivity_level": sensitivity_level,
        "masked_field_names": matched_pii_fields,
    }


# ==============================================================================
# Complete Phase 6 Validation Pipeline
# ==============================================================================

def run_validation_pipeline(
    doc_type: str,
    extracted_fields: dict[str, Any],
    tables: list[dict[str, Any]],
    ocr_results: dict[str, Any],
    corrected_image: np.ndarray,
    existing_tamper_flags: list[dict[str, Any]],
) -> dict[str, Any]:
    """Execute complete Phase 6 validation layer.

    Orchestrates business cross-checks, font-consistency tamper analysis,
    appends new font flags to existing tamper flags, and classifies confidentiality.

    Args:
        doc_type: Document classification category.
        extracted_fields: Phase 4/5 extracted fields dictionary.
        tables: Phase 4/5 extracted tables list.
        ocr_results: Complete Phase 2 OCR dictionary.
        corrected_image: Preprocessed/deskewed OpenCV image.
        existing_tamper_flags: List of tamper flags from Phase 3.

    Returns:
        Dict conforming to Phase 6 schema:
        {
            "cross_checks": [...],
            "tamper_flags": [...combined existing + font_inconsistency flags...],
            "confidentiality": {...}
        }
    """
    # 1. Business Logic Cross-Checks
    cross_checks = run_all_cross_checks(
        doc_type=doc_type,
        extracted_fields=extracted_fields,
        tables=tables,
    )

    # 2. Font Consistency Tamper Analysis
    font_flags = analyze_font_consistency(
        ocr_results=ocr_results,
        extracted_fields=extracted_fields,
        corrected_image=corrected_image,
    )

    # 3. Combine tamper flags (APPEND, never overwrite Phase 3 flags)
    combined_tamper_flags = list(existing_tamper_flags or [])
    combined_tamper_flags.extend(font_flags)

    # 4. Confidentiality & PII Classification
    confidentiality = classify_confidentiality(
        doc_type=doc_type,
        extracted_fields=extracted_fields,
    )

    return {
        "cross_checks": cross_checks,
        "tamper_flags": combined_tamper_flags,
        "confidentiality": confidentiality,
    }
