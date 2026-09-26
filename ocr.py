"""OCR and Layout Extraction Module - Phase 2: OCR + Layout Extraction.

This module provides standalone, independently callable functions for:
- Per-word text, confidence, and bounding box extraction using PyTesseract
- Grouping words into line-level bounding boxes and aggregated confidences
- Multi-page document OCR orchestration adhering to the Phase 2 JSON schema
- Visual bounding box overlay generation color-coded by confidence.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Any

import cv2
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# ==============================================================================
# Tesseract OCR Binary Resolution & Configuration Instructions
# ==============================================================================
# On Windows, if Tesseract-OCR is installed but not added to your system PATH,
# pytesseract will raise:
#   pytesseract.pytesseract.TesseractNotFoundError
#
# To manually configure the binary path if auto-detection fails, set:
#
#   import pytesseract
#   pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
#
# Alternatively, add C:\Program Files\Tesseract-OCR to your Windows PATH environment variable.
# ==============================================================================

import pytesseract

def _configure_tesseract() -> bool:
    """Detect and configure Tesseract executable path if not already on PATH."""
    # 1. Check if default command works
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        pass

    # 2. Check standard Windows installation directories
    candidate_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
    ]

    for path in candidate_paths:
        if os.path.isfile(path):
            pytesseract.pytesseract.tesseract_cmd = path
            try:
                pytesseract.get_tesseract_version()
                logger.info(f"Configured Tesseract binary at: {path}")
                return True
            except Exception:
                continue

    logger.warning(
        "Tesseract binary could not be found automatically. "
        "Please ensure Tesseract is installed and set pytesseract.pytesseract.tesseract_cmd."
    )
    return False

TESSERACT_AVAILABLE = _configure_tesseract()

# ==============================================================================
# Named Constants & Confidence Thresholds
# ==============================================================================
# Threshold below which a word is counted in low_confidence_word_count
LOW_CONFIDENCE_THRESHOLD: float = 0.60

# Threshold for high-confidence display
HIGH_CONFIDENCE_THRESHOLD: float = 0.80


def run_ocr_on_image(image: np.ndarray) -> dict[str, list[dict[str, Any]]]:
    """Run Tesseract OCR on a single image and extract per-word data.

    Uses pytesseract.image_to_data to obtain bounding boxes, confidence scores,
    and hierarchical layout positions (block, paragraph, line) in a single pass.
    Tesseract returns confidence as 0-100 and -1 for non-text regions.
    Confidence is scaled to 0-1 and non-text/empty entries are filtered out.

    Args:
        image: RGB numpy array representing a document page.

    Returns:
        Dict containing:
            "words": list of dicts with:
                - text: string
                - confidence: float (0.0 to 1.0)
                - bbox: {"x": int, "y": int, "width": int, "height": int}
                - _line_key: tuple(block_num, par_num, line_num) for layout grouping
    """
    if not TESSERACT_AVAILABLE and not _configure_tesseract():
        logger.error("Tesseract OCR is not available on this system.")
        return {"words": []}

    try:
        # Convert RGB to grayscale or BGR as needed by pytesseract
        # pytesseract handles RGB numpy arrays or PIL Images directly
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    except Exception as e:
        logger.error(f"Error executing pytesseract.image_to_data: {e}", exc_info=True)
        return {"words": []}

    words: list[dict[str, Any]] = []
    num_entries = len(data.get("text", []))

    for i in range(num_entries):
        raw_text = str(data["text"][i]).strip()
        raw_conf = float(data["conf"][i])

        # Exclude empty text or non-text markers (conf == -1)
        if not raw_text or raw_conf < 0:
            continue

        conf_scaled = round(float(raw_conf) / 100.0, 4)
        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])

        line_key = (
            int(data.get("block_num", [0] * num_entries)[i]),
            int(data.get("par_num", [0] * num_entries)[i]),
            int(data.get("line_num", [0] * num_entries)[i]),
        )

        words.append({
            "text": raw_text,
            "confidence": conf_scaled,
            "bbox": {"x": x, "y": y, "width": w, "height": h},
            "_line_key": line_key,
        })

    return {"words": words}


def group_words_into_lines(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group word-level OCR results into line-level entries.

    Utilizes the layout grouping keys (block_num, par_num, line_num) provided
    by Tesseract's image_to_data. Each line's confidence is the average of its
    words' confidence, and its bounding box is the spatial union of its words' bboxes.

    Args:
        words: List of word dicts containing text, confidence, bbox, and optionally _line_key.

    Returns:
        List of line dicts matching schema:
            - text: string (words joined by spaces)
            - confidence: float 0-1 (mean confidence of constituent words)
            - bbox: {"x": int, "y": int, "width": int, "height": int} (union bbox)
    """
    if not words:
        return []

    # Group words by their line key while preserving natural reading order
    lines_dict: dict[Any, list[dict[str, Any]]] = {}
    for w in words:
        key = w.get("_line_key")
        if key is None:
            # Fallback grouping by vertical proximity if _line_key is absent
            y_approx = w["bbox"]["y"] // 15
            key = (0, 0, y_approx)
        if key not in lines_dict:
            lines_dict[key] = []
        lines_dict[key].append(w)

    lines: list[dict[str, Any]] = []
    for line_words in lines_dict.values():
        if not line_words:
            continue

        # Sort words in line horizontally by left x-coordinate
        sorted_words = sorted(line_words, key=lambda item: item["bbox"]["x"])

        line_text = " ".join(item["text"] for item in sorted_words)
        avg_confidence = round(
            sum(item["confidence"] for item in sorted_words) / len(sorted_words),
            4,
        )

        min_x = min(item["bbox"]["x"] for item in sorted_words)
        min_y = min(item["bbox"]["y"] for item in sorted_words)
        max_x = max(item["bbox"]["x"] + item["bbox"]["width"] for item in sorted_words)
        max_y = max(item["bbox"]["y"] + item["bbox"]["height"] for item in sorted_words)

        union_bbox = {
            "x": int(min_x),
            "y": int(min_y),
            "width": int(max_x - min_x),
            "height": int(max_y - min_y),
        }

        lines.append({
            "text": line_text,
            "confidence": avg_confidence,
            "bbox": union_bbox,
        })

    return lines


def run_ocr_on_document(corrected_images: list[np.ndarray]) -> dict[str, Any]:
    """Execute OCR across all corrected document pages and construct Phase 2 output.

    Iterates through each page, extracts word and line entities, computes full
    text per page, and aggregates overall document confidence and low-confidence
    word counts.

    If OCR fails on a page (e.g. Tesseract missing or error), the page entry is
    returned with empty words/lines and full_text = "", surfacing a fail-loud
    condition for downstream consumers.

    Args:
        corrected_images: List of RGB numpy images (one per page) from Phase 1.

    Returns:
        Dict matching Phase 2 schema:
        {
          "ocr_results": {
            "pages": [
              {
                "page_number": int,
                "full_text": str,
                "words": [ {"text": str, "confidence": float, "bbox": dict} ],
                "lines": [ {"text": str, "confidence": float, "bbox": dict} ]
              }
            ],
            "average_confidence": float,
            "low_confidence_word_count": int
          }
        }
    """
    pages_output: list[dict[str, Any]] = []
    all_words: list[dict[str, Any]] = []

    for page_idx, img in enumerate(corrected_images):
        page_num = page_idx + 1
        page_failed = False
        words: list[dict[str, Any]] = []
        lines: list[dict[str, Any]] = []
        full_text = ""

        try:
            ocr_res = run_ocr_on_image(img)
            raw_words = ocr_res.get("words", [])
            lines = group_words_into_lines(raw_words)

            # Build full page text from lines
            full_text = "\n".join(l["text"] for l in lines)

            # Clean internal _line_key from words for the public schema
            clean_words: list[dict[str, Any]] = []
            for w in raw_words:
                clean_words.append({
                    "text": w["text"],
                    "confidence": w["confidence"],
                    "bbox": w["bbox"],
                })

            words = clean_words
            all_words.extend(words)

            if not words and img.size > 0:
                logger.warning(f"Page {page_num} produced zero recognized OCR words.")

        except Exception as e:
            logger.error(f"OCR failed entirely on page {page_num}: {e}", exc_info=True)
            page_failed = True
            words = []
            lines = []
            full_text = ""

        pages_output.append({
            "page_number": page_num,
            "full_text": full_text,
            "words": words,
            "lines": lines,
        })

    # Overall document metrics
    if all_words:
        avg_conf = round(float(sum(w["confidence"] for w in all_words) / len(all_words)), 4)
        low_conf_count = int(sum(1 for w in all_words if w["confidence"] < LOW_CONFIDENCE_THRESHOLD))
    else:
        avg_conf = 0.0
        low_conf_count = 0

    return {
        "ocr_results": {
            "pages": pages_output,
            "average_confidence": avg_conf,
            "low_confidence_word_count": low_conf_count,
        }
    }


def draw_ocr_bounding_boxes(
    image: np.ndarray,
    words: list[dict[str, Any]],
    low_thresh: float = LOW_CONFIDENCE_THRESHOLD,
    high_thresh: float = HIGH_CONFIDENCE_THRESHOLD,
) -> np.ndarray:
    """Draw confidence-color-coded bounding boxes on a page image.

    Colors:
        - Green (>= high_thresh): High confidence recognition
        - Amber (>= low_thresh and < high_thresh): Medium confidence recognition
        - Red (< low_thresh): Low confidence recognition

    Args:
        image: RGB page image as numpy array.
        words: List of word dicts with 'bbox' and 'confidence'.
        low_thresh: Low confidence cutoff threshold.
        high_thresh: High confidence cutoff threshold.

    Returns:
        New RGB image with annotated bounding boxes.
    """
    annotated = image.copy()

    for item in words:
        bbox = item.get("bbox", {})
        x = bbox.get("x", 0)
        y = bbox.get("y", 0)
        w = bbox.get("width", 0)
        h = bbox.get("height", 0)
        conf = float(item.get("confidence", 0.0))

        if conf >= high_thresh:
            color = (34, 197, 94)    # Modern Green
        elif conf >= low_thresh:
            color = (234, 179, 8)    # Amber / Yellow
        else:
            color = (239, 68, 68)    # Vibrant Red

        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

    return annotated
