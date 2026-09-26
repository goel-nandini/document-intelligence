"""Document Intelligence - Phases 1 & 2: Ingestion, Preprocessing & OCR.

Streamlit application providing:
- Document upload (PDF, PNG, JPG, JPEG)
- Metadata extraction & SHA-256 computation
- Blur & skew quality assessment with circuit breaker
- Side-by-side visual inspection (original vs deskewed/denoised)
- Automated Phase 2 OCR + Layout Extraction via Tesseract
- Interactive visual bounding box overlay color-coded by confidence
- Extracted text inspection and strict schema verification
"""

from __future__ import annotations

import json
import streamlit as st

import ocr
import preprocessing

# Page configuration
st.set_page_config(
    page_title="DocIntelligence | Phase 1 & 2",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1E293B;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .phase-badge {
        display: inline-block;
        background-color: #EEF2FF;
        color: #4F46E5;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        margin-bottom: 0.5rem;
    }
    .legend-box {
        display: flex;
        gap: 1.5rem;
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 0.6rem 1rem;
        margin-bottom: 1rem;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar with system controls & thresholds
with st.sidebar:
    st.markdown('<span class="phase-badge">Phase 1 & Phase 2</span>', unsafe_allow_html=True)
    st.title("System Controls")
    st.markdown(
        """
        **Active Pipelines:**
        - **Phase 1:** Ingestion, EXIF/PDF Metadata, Blur & Skew Assessment, Deskewing, Bilateral Denoising.
        - **Phase 2:** Tesseract OCR, Line Grouping, Spatial Bounding Boxes, Confidence Calibration.
        """
    )
    st.divider()
    st.markdown(f"**Blur Threshold:** `{preprocessing.BLUR_THRESHOLD}`")
    st.markdown(f"**Low Confidence Cutoff:** `{ocr.LOW_CONFIDENCE_THRESHOLD * 100:.0f}%`")
    st.markdown(f"**High Confidence Cutoff:** `{ocr.HIGH_CONFIDENCE_THRESHOLD * 100:.0f}%`")
    st.divider()
    if ocr.TESSERACT_AVAILABLE:
        st.success("Tesseract OCR Engine: Online")
    else:
        st.error("Tesseract OCR Engine: Offline (Check PATH or tesseract_cmd)")
    st.caption("Document Intelligence Engine v2.0")

# Header Section
st.markdown('<span class="phase-badge">Document Trust Layer</span>', unsafe_allow_html=True)
st.markdown('<div class="main-header">Document Intelligence System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Upload a document to run ingestion, automated quality enhancement, and spatial OCR layout extraction.</div>',
    unsafe_allow_html=True,
)

# File Uploader
uploaded_file = st.file_uploader(
    "Choose a document to ingest",
    type=["pdf", "png", "jpg", "jpeg"],
    help="Supported formats: PDF (multi-page supported), PNG, JPG, JPEG",
)

if uploaded_file is not None:
    try:
        # =============================================================
        # Phase 1: Ingestion & Preprocessing
        # =============================================================
        with st.spinner("Executing Phase 1: Ingesting, assessing quality, deskewing & denoising..."):
            prep_result = preprocessing.preprocess_document(uploaded_file)

        metadata = prep_result["file_metadata"]
        quality = prep_result["quality_assessment"]
        original_images = prep_result["original_images"]
        corrected_images = prep_result["corrected_images"]
        page_assessments = prep_result.get("page_assessments", [])

        # 1. Document Metadata Display
        st.subheader("📋 Document Metadata")
        meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
        with meta_col1:
            st.metric("Filename", metadata["original_filename"])
        with meta_col2:
            st.metric("Total Pages", metadata["page_count"])
        with meta_col3:
            created = metadata["creation_date_from_metadata"] or "N/A"
            st.metric("Creation Date", created if len(created) <= 19 else created[:19])
        with meta_col4:
            software = metadata["software_used"] or "N/A"
            st.metric("Software Used", software if len(software) <= 20 else software[:17] + "...")

        with st.expander("🔍 Complete Metadata & SHA-256 Hash", expanded=False):
            st.markdown(f"**SHA-256 Hash:** `{metadata['file_hash_sha256']}`")
            st.markdown(f"**Modification Date:** `{metadata['modification_date_from_metadata'] or 'None'}`")
            st.markdown(f"**Software / Producer:** `{metadata['software_used'] or 'None'}`")

        # 2. Quality Assessment Display
        st.subheader("📊 Quality Assessment")
        qa_col1, qa_col2, qa_col3, qa_col4 = st.columns(4)
        with qa_col1:
            st.metric("Blur Score", f"{quality['blur_score']:.1f}")
        with qa_col2:
            st.metric("Blur Threshold", f"{preprocessing.BLUR_THRESHOLD:.1f}")
        with qa_col3:
            st.metric("Skew Angle", f"{quality['skew_angle_degrees']:.2f}°")
        with qa_col4:
            st.metric("Quality Status", quality["status"].upper())

        # Degraded Circuit Breaker: Halt immediately if rejected
        if quality["status"] == "rejected_too_degraded":
            st.error(
                f"🛑 **Document Rejected (Too Degraded):** Document quality assessment detected severe blur "
                f"(score {quality['blur_score']} is below threshold {preprocessing.BLUR_THRESHOLD}). "
                f"Ingestion has halted and downstream processing is blocked."
            )
            st.stop()

        # Success Banner
        st.success("✅ **Quality Assessment Passed:** Document is clear and meets processing standards.")

        # 3. Visual Inspection: Original vs Corrected
        with st.expander("🖼️ Visual Inspection: Original vs Corrected Pages", expanded=False):
            num_pages = len(original_images)
            for page_idx in range(num_pages):
                orig_page_img = original_images[page_idx]
                corr_page_img = corrected_images[page_idx]
                page_info = page_assessments[page_idx] if page_idx < len(page_assessments) else None

                if num_pages > 1:
                    st.markdown(f"**Page {page_idx + 1} of {num_pages}**")

                col_orig, col_corr = st.columns(2)
                with col_orig:
                    st.caption(
                        f"**Original Page {page_idx + 1}** — "
                        f"Blur: `{page_info['original_qa']['blur_score'] if page_info else 'N/A'}` | "
                        f"Skew: `{page_info['original_qa']['skew_angle_degrees'] if page_info else 'N/A'}°`"
                    )
                    st.image(orig_page_img, use_container_width=True)

                with col_corr:
                    st.caption(
                        f"**Corrected Page {page_idx + 1} (Deskewed & Denoised)** — "
                        f"Blur: `{page_info['final_qa']['blur_score'] if page_info else 'N/A'}` | "
                        f"Skew: `{page_info['final_qa']['skew_angle_degrees'] if page_info else 'N/A'}°`"
                    )
                    st.image(corr_page_img, use_container_width=True)

        # 4. Phase 1 Schema Output Block
        with st.expander("📄 Phase 1 Output JSON Schema Contract", expanded=False):
            phase1_schema = {
                "document_id": prep_result["document_id"],
                "upload_timestamp": prep_result["upload_timestamp"],
                "file_metadata": metadata,
                "quality_assessment": quality,
            }
            st.json(phase1_schema)

        # =============================================================
        # Phase 2: OCR + Layout Extraction
        # =============================================================
        st.divider()
        st.subheader("📑 Phase 2: OCR & Layout Extraction")

        with st.spinner("Executing Phase 2: Running Tesseract OCR & layout extraction..."):
            ocr_output = ocr.run_ocr_on_document(corrected_images)

        ocr_results = ocr_output["ocr_results"]
        pages_ocr = ocr_results.get("pages", [])
        avg_confidence = ocr_results.get("average_confidence", 0.0)
        low_conf_count = ocr_results.get("low_confidence_word_count", 0)

        # Total recognized words
        total_words = sum(len(p.get("words", [])) for p in pages_ocr)

        # OCR Metrics Display
        ocr_col1, ocr_col2, ocr_col3, ocr_col4 = st.columns(4)
        with ocr_col1:
            st.metric("Avg OCR Confidence", f"{avg_confidence * 100:.1f}%")
        with ocr_col2:
            st.metric("Total Recognized Words", total_words)
        with ocr_col3:
            st.metric("Low Confidence Words", low_conf_count)
        with ocr_col4:
            st.metric("Low Conf Threshold", f"<{ocr.LOW_CONFIDENCE_THRESHOLD * 100:.0f}%")

        # Color Legend for Bounding Boxes
        st.markdown(
            """
            <div class="legend-box">
                <span><b>Bounding Box Legend:</b></span>
                <span style="color: #16A34A; font-weight: 600;">🟢 High Confidence (≥ 80%)</span>
                <span style="color: #D97706; font-weight: 600;">🟡 Medium Confidence (60% - 79%)</span>
                <span style="color: #DC2626; font-weight: 600;">🔴 Low Confidence (< 60%)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Visual Overlay & Extracted Text Display per page
        for page_idx, page_data in enumerate(pages_ocr):
            page_num = page_data.get("page_number", page_idx + 1)
            page_words = page_data.get("words", [])
            page_text = page_data.get("full_text", "")
            corr_img = corrected_images[page_idx] if page_idx < len(corrected_images) else None

            if len(pages_ocr) > 1:
                st.markdown(f"#### Page {page_num} of {len(pages_ocr)}")

            # Loud failure check: if page produced 0 words or empty text
            if not page_words and not page_text.strip():
                st.warning(
                    f"⚠️ **Fail Loud — Page {page_num} OCR Produced Zero Words:** "
                    f"No text could be extracted from this page. "
                    f"Please verify if the page is blank, corrupted, or if Tesseract encountered an unreadable font."
                )
                if corr_img is not None:
                    st.image(corr_img, caption=f"Page {page_num} (No OCR words recognized)", use_container_width=True)
                continue

            col_annotated, col_text = st.columns([1, 1])

            with col_annotated:
                st.markdown(f"**Bounding Box Alignment Overlay (Page {page_num})**")
                if corr_img is not None:
                    # Draw color-coded bounding boxes
                    annotated_img = ocr.draw_ocr_bounding_boxes(corr_img, page_words)
                    st.image(
                        annotated_img,
                        caption=f"Page {page_num} — {len(page_words)} words recognized",
                        use_container_width=True,
                    )
                else:
                    st.info("Image data unavailable for overlay.")

            with col_text:
                st.markdown(f"**Extracted Text (Page {page_num})**")
                st.text_area(
                    label=f"Full text extracted from page {page_num}",
                    value=page_text,
                    height=450,
                    key=f"ocr_text_page_{page_num}",
                )

            if page_idx < len(pages_ocr) - 1:
                st.divider()

        # -------------------------------------------------------------
        # Phase 2 JSON Schema Section
        # -------------------------------------------------------------
        st.subheader("📑 Phase 2 Schema Output")
        with st.expander("📄 Raw ocr_results JSON (Phase 2 Contract)", expanded=True):
            st.json(ocr_output)
            st.download_button(
                label="📥 Download Phase 2 ocr_results JSON",
                data=json.dumps(ocr_output, indent=2),
                file_name=f"document_{prep_result['document_id'][:8]}_phase2_ocr.json",
                mime="application/json",
            )

    except Exception as e:
        st.error(f"❌ **Processing Error:** An unexpected error occurred: {str(e)}")
        st.exception(e)
else:
    st.info("👆 Please upload a PDF, PNG, or JPG document above to begin Ingestion and OCR layout extraction.")
