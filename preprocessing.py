"""Document Preprocessing Module - Phase 1: Ingestion + Preprocessing.

This module provides standalone, independently callable functions for:
- File metadata extraction and SHA-256 computation
- Document loading (PDF, JPG, PNG) into RGB image representations
- Document image quality assessment (blur and skew detection)
- Image enhancement (deskewing and denoising)
- Document preprocessing orchestration adhering to the Phase 1 schema.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np
from PIL import ExifTags, Image

try:
    import pymupdf as fitz
except ImportError:
    import fitz

# Named threshold constant for blur assessment (Laplacian variance)
# Images with a blur score below this threshold are considered degraded.
BLUR_THRESHOLD: float = 100.0


def compute_file_hash(file_bytes: bytes) -> str:
    """Compute SHA-256 hash of raw file bytes.

    Args:
        file_bytes: Raw bytes of the document.

    Returns:
        Hexadecimal SHA-256 hash string.
    """
    return hashlib.sha256(file_bytes).hexdigest()


def _parse_pdf_date(date_str: str | None) -> str | None:
    """Parse PyMuPDF date string (e.g. 'D:20240115123000Z') to ISO-8601 string."""
    if not date_str:
        return None
    try:
        clean = date_str.replace("D:", "").replace("'", "")
        # Match YYYYMMDDHHMMSS or at least YYYYMMDD
        match = re.match(r"^(\d{4})(\d{2})(\d{2})(?:(\d{2})(\d{2})(\d{2}))?", clean)
        if match:
            year, month, day = match.group(1), match.group(2), match.group(3)
            hour = match.group(4) or "00"
            minute = match.group(5) or "00"
            second = match.group(6) or "00"
            dt = datetime(
                int(year),
                int(month),
                int(day),
                int(hour),
                int(minute),
                int(second),
                tzinfo=timezone.utc,
            )
            return dt.isoformat()
    except Exception:
        pass
    return None


def _parse_exif_date(date_str: str | None) -> str | None:
    """Parse EXIF date string (e.g. '2024:01:15 12:30:00') to ISO-8601 string."""
    if not date_str:
        return None
    try:
        # Standard EXIF format: "%Y:%m:%d %H:%M:%S"
        dt = datetime.strptime(str(date_str).strip(), "%Y:%m:%d %H:%M:%S")
        return dt.replace(tzinfo=timezone.utc).isoformat()
    except Exception:
        pass
    return None


def extract_file_metadata(file: Any, file_bytes: bytes) -> dict[str, Any]:
    """Extract metadata matching the file_metadata schema.

    For PDFs, creation_date, modification_date, and software_used are extracted
    from PyMuPDF doc.metadata. For images, EXIF data is extracted via PIL if present.
    If unavailable, those fields are set to null.

    Args:
        file: File object, file-like object, or filename string.
        file_bytes: Raw bytes of the document.

    Returns:
        Dict matching file_metadata schema:
            original_filename, file_hash_sha256, page_count,
            creation_date_from_metadata, modification_date_from_metadata,
            software_used.
    """
    filename = "unknown_document"
    if hasattr(file, "name") and file.name:
        filename = os.path.basename(file.name)
    elif isinstance(file, str):
        filename = os.path.basename(file)

    file_hash = compute_file_hash(file_bytes)
    is_pdf = filename.lower().endswith(".pdf") or file_bytes.startswith(b"%PDF")

    creation_date: str | None = None
    modification_date: str | None = None
    software_used: str | None = None
    page_count: int = 1

    if is_pdf:
        try:
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                page_count = len(doc)
                meta = doc.metadata or {}
                creation_date = _parse_pdf_date(meta.get("creationDate"))
                modification_date = _parse_pdf_date(meta.get("modDate"))
                software_used = meta.get("producer") or meta.get("creator") or None
        except Exception as e:
            # Fall back to default null values on metadata read failure
            pass
    else:
        # Image file metadata (EXIF)
        page_count = 1
        try:
            img = Image.open(io.BytesIO(file_bytes))
            exif_data = img.getexif()
            if exif_data:
                tag_map = {
                    ExifTags.TAGS.get(tag, tag): val
                    for tag, val in exif_data.items()
                }
                # Creation date (DateTimeOriginal or DateTimeDigitized or DateTime)
                creation_str = tag_map.get("DateTimeOriginal") or tag_map.get("DateTime")
                mod_str = tag_map.get("DateTime")
                creation_date = _parse_exif_date(creation_str)
                modification_date = _parse_exif_date(mod_str)
                software_used = tag_map.get("Software") or None
        except Exception:
            pass

    return {
        "original_filename": filename,
        "file_hash_sha256": file_hash,
        "page_count": page_count,
        "creation_date_from_metadata": creation_date,
        "modification_date_from_metadata": modification_date,
        "software_used": software_used,
    }


def load_document(file: Any) -> list[np.ndarray]:
    """Load document into a list of RGB numpy images (one per page).

    Supports PDF, JPG, PNG, and standard image formats.

    Args:
        file: Streamlit UploadedFile, file-like object, filepath string, or bytes.

    Returns:
        List of np.ndarray images in RGB format (shape: H x W x 3, dtype: uint8).
    """
    if isinstance(file, bytes):
        file_bytes = file
        filename = "document"
    elif hasattr(file, "read"):
        if hasattr(file, "seek"):
            file.seek(0)
        file_bytes = file.read()
        if hasattr(file, "seek"):
            file.seek(0)
        filename = getattr(file, "name", "document")
    elif isinstance(file, str):
        filename = os.path.basename(file)
        with open(file, "rb") as f:
            file_bytes = f.read()
    else:
        raise ValueError(f"Unsupported file input type: {type(file)}")

    is_pdf = filename.lower().endswith(".pdf") or file_bytes.startswith(b"%PDF")
    images: list[np.ndarray] = []

    if is_pdf:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page_index in range(len(doc)):
                page = doc[page_index]
                # Render at 150 DPI for optimal document balance between fidelity and speed
                zoom = 150.0 / 72.0
                matrix = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                # Convert pixmap buffer to numpy array
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
                if pix.n == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif pix.n == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                elif pix.n == 3:
                    img = img.copy()
                images.append(img)
    else:
        pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        images.append(np.array(pil_img, dtype=np.uint8))

    return images


def detect_blur(image: np.ndarray) -> float:
    """Compute blur score using Laplacian variance of grayscale image.

    Higher values indicate sharp edges; lower values indicate blurriness.

    Args:
        image: RGB or grayscale image as numpy array.

    Returns:
        Laplacian variance as a float.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def detect_skew_angle(image: np.ndarray) -> float:
    """Detect skew angle of text lines in degrees using Hough line transform.

    Args:
        image: RGB or grayscale image as numpy array.

    Returns:
        Detected skew angle in degrees (positive or negative).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Binarize with Otsu's thresholding (invert so text is white)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilate horizontally to connect letters into continuous text lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 2))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # Canny edge detection on text lines
    edges = cv2.Canny(dilated, 50, 150, apertureSize=3)

    # Probabilistic Hough Line detection
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=60, maxLineGap=20)

    angles: list[float] = []
    if lines is not None:
        lines = lines.reshape(-1, 4)
        for x1, y1, x2, y2 in lines:
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            if dx == 0:
                continue
            angle = float(np.degrees(np.arctan2(dy, dx)))
            # Keep plausible skew angles within [-45, 45] degrees
            if -45.0 < angle < 45.0:
                angles.append(angle)

    if angles:
        # Return median angle to reject outliers from borders or figures
        return float(np.median(angles))

    # Fallback to contour minAreaRect
    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_angles: list[float] = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 300:
            rect = cv2.minAreaRect(cnt)
            angle = rect[-1]
            w, h = rect[1]
            if w < h:
                angle = angle - 90.0 if angle > 0 else angle + 90.0
            if -45.0 < angle < 45.0 and abs(angle) > 0.1:
                contour_angles.append(angle)

    if contour_angles:
        return float(np.median(contour_angles))

    return 0.0


def deskew_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by the given angle to correct skew.

    Fills expanded border areas with white (255) to blend seamlessly with document backgrounds.

    Args:
        image: RGB image as numpy array.
        angle: Skew angle in degrees.

    Returns:
        Deskewed image as numpy array.
    """
    if abs(angle) < 0.1:
        return image.copy()

    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # cv2.getRotationMatrix2D rotates counterclockwise for positive angle.
    # Since dy > 0 represents clockwise tilt in image coordinates (y increases downward),
    # rotating counter-clockwise by +angle levels it back to horizontal.
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # White background for borders
    border_color = (255, 255, 255) if len(image.shape) == 3 else 255

    deskewed = cv2.warpAffine(
        image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_color,
    )
    return deskewed


def denoise_image(image: np.ndarray) -> np.ndarray:
    """Denoise document image using bilateral filtering.

    Preserves crisp document text and sharp edges while smoothing background grain/noise.

    Args:
        image: RGB image as numpy array.

    Returns:
        Denoised RGB image.
    """
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)


def assess_quality(image: np.ndarray) -> dict[str, Any]:
    """Assess document image quality against blur and skew thresholds.

    Args:
        image: RGB or grayscale image as numpy array.

    Returns:
        Dict matching quality_assessment schema:
            blur_score: float
            skew_angle_degrees: float
            is_too_degraded: boolean
            status: 'ok' | 'rejected_too_degraded'
    """
    blur = detect_blur(image)
    skew = detect_skew_angle(image)
    is_too_degraded = bool(blur < BLUR_THRESHOLD)
    status = "rejected_too_degraded" if is_too_degraded else "ok"

    return {
        "blur_score": round(float(blur), 2),
        "skew_angle_degrees": round(float(skew), 2),
        "is_too_degraded": is_too_degraded,
        "status": status,
    }


def preprocess_document(file: Any) -> dict[str, Any]:
    """Ingest, inspect, assess, and preprocess a document.

    Orchestrates the entire Phase 1 pipeline:
    1. Generates document_id and upload_timestamp.
    2. Extracts file_metadata from raw bytes.
    3. Loads document into page images.
    4. Runs quality assessment on original pages.
    5. Deskews and denoises each page.
    6. Runs quality assessment on corrected pages.
    7. Formulates overall document quality assessment:
       For multi-page documents, if ANY page has is_too_degraded == True,
       the overall document status is marked as 'rejected_too_degraded'.
       (Design Decision Note: In document intelligence workflows such as KYC,
        invoices, or contract processing, a single illegible or corrupted page
        can invalidate the entire document transaction or downstream extraction.
        If your workflow prefers per-page salvage/partial acceptance, you can
        modify this aggregation to average blur_score or track per-page status).

    Args:
        file: Streamlit UploadedFile, file-like object, or filepath.

    Returns:
        Dict containing:
            - document_id (str)
            - upload_timestamp (str, ISO-8601)
            - file_metadata (dict)
            - quality_assessment (dict, evaluated on final corrected pages)
            - original_images (list[np.ndarray], internal pipeline key)
            - corrected_images (list[np.ndarray], internal pipeline key)
            - page_assessments (list[dict], per-page before/after metrics for UI/audit)
    """
    # 1. Read file bytes
    if isinstance(file, bytes):
        file_bytes = file
    elif hasattr(file, "read"):
        if hasattr(file, "seek"):
            file.seek(0)
        file_bytes = file.read()
        if hasattr(file, "seek"):
            file.seek(0)
    elif isinstance(file, str):
        with open(file, "rb") as f:
            file_bytes = f.read()
    else:
        raise ValueError(f"Unsupported file type: {type(file)}")

    # 2. Document Identifiers
    doc_id = str(uuid.uuid4())
    upload_ts = datetime.now(timezone.utc).isoformat()

    # 3. Metadata Extraction
    metadata = extract_file_metadata(file, file_bytes)

    # 4. Load Document Images
    original_images = load_document(file)
    if not original_images:
        raise ValueError("No readable pages or images could be loaded from the document.")

    corrected_images: list[np.ndarray] = []
    page_assessments: list[dict[str, Any]] = []

    # 5. Process Each Page
    for page_idx, orig_img in enumerate(original_images):
        initial_qa = assess_quality(orig_img)

        # Deskew and Denoise
        skew_angle = initial_qa["skew_angle_degrees"]
        deskewed_img = deskew_image(orig_img, skew_angle)
        corrected_img = denoise_image(deskewed_img)
        corrected_images.append(corrected_img)

        # Re-assess quality after corrections
        final_qa = assess_quality(corrected_img)

        page_assessments.append({
            "page_number": page_idx + 1,
            "original_qa": initial_qa,
            "final_qa": final_qa,
        })

    # Multi-page aggregation:
    # Set overall status to "rejected_too_degraded" if ANY page is too degraded.
    any_page_degraded = any(p["final_qa"]["is_too_degraded"] for p in page_assessments)
    overall_status = "rejected_too_degraded" if any_page_degraded else "ok"

    # Aggregated blur and skew scores across pages (worst-case blur, max absolute skew)
    min_blur = min(p["final_qa"]["blur_score"] for p in page_assessments)
    max_skew = max((p["final_qa"]["skew_angle_degrees"] for p in page_assessments), key=abs)

    final_quality_assessment = {
        "blur_score": round(float(min_blur), 2),
        "skew_angle_degrees": round(float(max_skew), 2),
        "is_too_degraded": any_page_degraded,
        "status": overall_status,
    }

    return {
        # Strict Schema Keys:
        "document_id": doc_id,
        "upload_timestamp": upload_ts,
        "file_metadata": metadata,
        "quality_assessment": final_quality_assessment,
        # Pipeline Data Keys (for UI and Phase 2, separate from schema):
        "original_images": original_images,
        "corrected_images": corrected_images,
        "page_assessments": page_assessments,
    }
