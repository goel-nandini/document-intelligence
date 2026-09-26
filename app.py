"""Document Intelligence - Phases 1, 2, 3 & 4.

Streamlit application providing:
- Phase 1: Ingestion, Metadata Extraction, Quality Assessment & Image Correction
- Phase 3: Exact Hash & Perceptual Hash Deduplication, Authenticity Pre-Checks (Metadata & ELA)
- Phase 2: Spatial Layout OCR via Tesseract, Confidence-calibrated Bounding Boxes, Full Text
- Phase 4: Schema-Driven Field & Table Extraction via Google Gemini, Spatial Grounding & DB Caching
"""

from __future__ import annotations

import json
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import authenticity_check
import db
import extraction
import ocr
import preprocessing

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="DocIntelligence | Trust Layer",
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
    .doc-type-pill {
        display: inline-block;
        background-color: #0F172A;
        color: #F8FAFC;
        font-size: 1.1rem;
        font-weight: 700;
        padding: 0.35rem 1rem;
        border-radius: 8px;
        letter-spacing: 0.05em;
        margin-bottom: 1rem;
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

# Sidebar with system controls & API configuration
with st.sidebar:
    st.markdown('<span class="phase-badge">Phases 1 - 4 Active</span>', unsafe_allow_html=True)
    st.title("System Controls")

    # Google Gemini API Key
    gemini_key_env = os.environ.get("GEMINI_API_KEY", "")

    if gemini_key_env:
        st.success("🟢 Google Gemini API: Active (.env)")
    else:
        st.warning("⚠️ No GEMINI_API_KEY detected in .env")

    with st.expander("🔑 Gemini API Key Settings", expanded=not bool(gemini_key_env)):
        gemini_input = st.text_input(
            "Gemini API Key",
            value=gemini_key_env,
            type="password",
            help="Used for Phase 4 field extraction via Google Gemini.",
        )
        if gemini_input:
            os.environ["GEMINI_API_KEY"] = gemini_input.strip()

    st.markdown(
        """
        **Active Pipelines:**
        - **Phase 1:** Ingestion, SHA-256 Hashing, EXIF/PDF Metadata, Blur & Skew Assessment, Deskewing, Bilateral Denoising.
        - **Phase 3:** Exact Hash & pHash Deduplication (Hamming dist <= 5), Timeline Inconsistency, Suspicious Software Checks, Error Level Analysis (ELA).
        - **Phase 2:** Tesseract OCR, Line Grouping, Spatial Bounding Boxes, Confidence Calibration.
        - **Phase 4:** Google Gemini Document Classification, Schema-Driven Field Extraction, Table Extraction & Spatial Bbox Grounding.
        """
    )
    st.divider()
    st.markdown(f"**Blur Threshold:** `{preprocessing.BLUR_THRESHOLD}`")
    st.markdown(f"**Low Confidence OCR Cutoff:** `{ocr.LOW_CONFIDENCE_THRESHOLD * 100:.0f}%`")
    st.markdown(f"**Metadata Time Gap Threshold:** `{authenticity_check.METADATA_TIME_GAP_THRESHOLD_DAYS} day(s)`")
    st.divider()
    if ocr.TESSERACT_AVAILABLE:
        st.success("Tesseract OCR: Online")
    else:
        st.error("Tesseract OCR: Offline (Check PATH or tesseract_cmd)")
    st.caption("Document Intelligence Engine v4.0")

# Header Section
st.markdown('<span class="phase-badge">Document Trust & Extraction Layer</span>', unsafe_allow_html=True)
st.markdown('<div class="main-header">Document Intelligence System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Automated ingestion, deduplication, authenticity pre-check, OCR layout, and schema-driven field extraction.</div>',
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
        with st.spinner("Phase 1: Ingesting, extracting metadata, deskewing & denoising..."):
            prep_result = preprocessing.preprocess_document(uploaded_file)

        doc_id = prep_result["document_id"]
        upload_ts = prep_result["upload_timestamp"]
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

        with st.expander("🔍 Complete Metadata & File Hash", expanded=False):
            st.markdown(f"**Document ID:** `{doc_id}`")
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
        st.success("✅ **Quality Assessment Passed:** Document meets quality processing standards.")

        # Side-by-side Visual Inspection (Phase 1)
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

        # =============================================================
        # Phase 3: Duplicate Detection + Authenticity Pre-Check
        # =============================================================
        st.divider()
        st.subheader("🛡️ Phase 3: Duplicate Detection & Authenticity Pre-Check")

        # Compute perceptual hash of first corrected image
        perceptual_hash = authenticity_check.compute_perceptual_hash(corrected_images[0])

        # Check for duplicates in SQLite database
        duplicate_match = db.find_duplicate(
            file_hash=metadata["file_hash_sha256"],
            perceptual_hash=perceptual_hash,
            similarity_threshold=5,
        )

        # -------------------------------------------------------------
        # IF DUPLICATE DETECTED: Fast-path return cached result & stop
        # -------------------------------------------------------------
        if duplicate_match is not None:
            matched_id = duplicate_match["matched_document_id"]
            match_type = duplicate_match["match_type"]

            st.warning(
                f"⚡ **Duplicate detected (match type: `{match_type}`)** — "
                f"Showing cached result instantly from matched document `{matched_id}`."
            )

            # Phase 3 Validation schema block for duplicate
            validation_schema = {
                "validation": {
                    "duplication_check": {
                        "is_duplicate": True,
                        "matched_document_id": matched_id,
                        "match_type": match_type,
                    },
                    "tamper_flags": [],
                }
            }

            st.json(validation_schema)

            # Fetch and display cached pipeline result if available
            cached_result = db.get_cached_result(matched_id)
            if cached_result:
                st.subheader("📦 Cached Pipeline Result")
                if "doc_type" in cached_result:
                    st.markdown(
                        f'<div class="doc-type-pill">CACHED DOC TYPE: {cached_result["doc_type"].upper()}</div>',
                        unsafe_allow_html=True,
                    )
                with st.expander("📄 Full Cached Result JSON", expanded=True):
                    st.json(cached_result)

                # Show cached fields if present
                if "extracted_fields" in cached_result:
                    st.markdown("**Cached Extracted Fields:**")
                    field_rows = [
                        {"Field Name": k, "Extracted Value": v.get("value"), "Raw OCR Text": v.get("raw_ocr_text")}
                        for k, v in cached_result["extracted_fields"].items()
                    ]
                    if field_rows:
                        st.dataframe(pd.DataFrame(field_rows), use_container_width=True)
            else:
                st.info(
                    f"A record for document `{matched_id}` is registered in the database. "
                    "Full downstream extraction has not yet been cached."
                )

            # Circuit breaker: Stop immediately, do not run Phase 2/4 again
            st.stop()

        # -------------------------------------------------------------
        # IF NOT A DUPLICATE: Run Authenticity Pre-check & Proceed
        # -------------------------------------------------------------
        st.success("✅ **Duplication Check Passed:** Document is original (no duplicate found in repository).")

        with st.spinner("Phase 3: Inspecting metadata timeline & Error Level Analysis (ELA)..."):
            tamper_flags = authenticity_check.run_authenticity_precheck(
                image=corrected_images[0],
                file_metadata=metadata,
            )

        # Build validation schema dictionary
        validation_schema = {
            "validation": {
                "duplication_check": {
                    "is_duplicate": False,
                    "matched_document_id": None,
                    "match_type": None,
                },
                "tamper_flags": tamper_flags,
            }
        }

        # Immediately display tamper flags as prominent red/yellow boxes (never hidden)
        if tamper_flags:
            st.markdown("#### ⚠️ Authenticity & Tamper Alerts")
            for flag in tamper_flags:
                f_type = flag["flag_type"]
                f_sev = flag["severity"]
                f_ev = flag["evidence"]

                if f_sev == "high":
                    st.error(f"🚨 **Tamper Flag [{f_type.upper()}] — HIGH Severity**\n\n{f_ev}")
                elif f_sev == "medium":
                    st.warning(f"⚠️ **Authenticity Flag [{f_type.upper()}] — MEDIUM Severity**\n\n{f_ev}")
                else:
                    st.info(f"ℹ️ **Observation [{f_type.upper()}] — LOW Severity**\n\n{f_ev}")
        else:
            st.success("✅ **Authenticity Pre-Check Passed:** No metadata inconsistencies or recompression anomalies detected.")

        # Save initial document record to database (result_json=None until extraction completes)
        db.save_document_record(
            document_id=doc_id,
            file_hash=metadata["file_hash_sha256"],
            perceptual_hash=perceptual_hash,
            upload_timestamp=upload_ts,
            result_json=None,
        )

        with st.expander("📄 Phase 3 Validation Schema Contract", expanded=False):
            st.json(validation_schema)

        # =============================================================
        # Phase 2: OCR + Layout Extraction
        # =============================================================
        st.divider()
        st.subheader("📑 Phase 2: OCR & Layout Extraction")

        with st.spinner("Phase 2: Running Tesseract OCR & spatial layout extraction..."):
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

        # =============================================================
        # Phase 4: Schema-Driven Field & Table Extraction
        # =============================================================
        st.divider()
        st.subheader("🧠 Phase 4: Schema-Driven Field & Table Extraction")

        # Execute extraction pipeline with clear error handling for Gemini
        try:
            with st.spinner("Phase 4: Classifying document type and extracting schema fields via Google Gemini..."):
                extraction_result = extraction.run_extraction_pipeline(ocr_results)
        except Exception as e:
            st.error(f"🛑 **Extraction Failed — Could not reach Google Gemini service:** {str(e)}")
            st.info(
                "💡 Please verify that your `GEMINI_API_KEY` is configured in `.env` "
                "or entered into the sidebar controls."
            )
            st.stop()

        detected_doc_type = extraction_result["doc_type"]
        extracted_fields = extraction_result.get("extracted_fields", {})
        extracted_tables = extraction_result.get("tables", [])
        skipped_fields = extraction_result.get("_skipped_fields", [])
        unmapped_bbox_fields = extraction_result.get("_unmapped_bbox_fields", [])

        # 1. Prominent Document Type Display
        st.markdown(
            f'<div class="doc-type-pill">DOCUMENT CLASSIFICATION: {detected_doc_type.upper()}</div>',
            unsafe_allow_html=True,
        )

        # 2. Warnings for omitted / unfound fields
        if skipped_fields:
            st.warning(
                f"⚠️ **Omitted / Unfound Fields:** The following fields from the `{detected_doc_type}` schema "
                f"were not found in the document text (LLM returned null and they were excluded): "
                f"{', '.join(f'`{f}`' for f in skipped_fields)}"
            )

        # 3. Warnings for fields without spatial bounding box
        if unmapped_bbox_fields:
            st.warning(
                f"⚠️ **Spatial Grounding Warning:** Bounding boxes could not be located in OCR text for: "
                f"{', '.join(f'`{f}`' for f in unmapped_bbox_fields)} (values exist but lack source region coordinates)."
            )

        # 4. Render Extracted Fields Table
        st.markdown("#### 📌 Extracted Key-Value Fields")
        if extracted_fields:
            field_table_data = []
            for fname, finfo in extracted_fields.items():
                bbox = finfo.get("bbox")
                if bbox:
                    bbox_str = f"Page {bbox.get('page', 1)} [{bbox.get('x')}, {bbox.get('y')}, {bbox.get('width')}x{bbox.get('height')}]"
                else:
                    bbox_str = "⚠️ Not Located"

                field_table_data.append({
                    "Field Name": fname,
                    "Extracted Value": finfo.get("value"),
                    "Raw OCR Text": finfo.get("raw_ocr_text"),
                    "Field Type": finfo.get("field_type"),
                    "Source Bounding Box": bbox_str,
                })
            st.dataframe(pd.DataFrame(field_table_data), use_container_width=True)
        else:
            st.info("No key-value fields were extracted from this document.")

        # 5. Render Tables (e.g. line_items, deductions, transactions)
        if extracted_tables:
            st.markdown("#### 📊 Extracted Tables")
            for t in extracted_tables:
                t_name = t.get("table_name", "table")
                rows = t.get("rows", [])
                st.markdown(f"**Table: `{t_name}`** ({len(rows)} rows)")

                if rows:
                    flattened_rows = []
                    for r in rows:
                        row_dict = {"Row #": r.get("row_index")}
                        cells = r.get("cells", {})
                        for col_name, c_data in cells.items():
                            row_dict[col_name] = c_data.get("value")
                        flattened_rows.append(row_dict)
                    st.dataframe(pd.DataFrame(flattened_rows), use_container_width=True)
                else:
                    st.caption("Table structure was detected but contains no data rows.")

        # 6. Update SQLite Cache with Complete Pipeline Results
        complete_pipeline_result = {
            "document_id": doc_id,
            "upload_timestamp": upload_ts,
            "file_metadata": metadata,
            "quality_assessment": quality,
            "validation": validation_schema["validation"],
            "ocr_results": ocr_results,
            "doc_type": detected_doc_type,
            "extracted_fields": extracted_fields,
            "tables": extracted_tables,
        }
        db.save_document_record(
            document_id=doc_id,
            file_hash=metadata["file_hash_sha256"],
            perceptual_hash=perceptual_hash,
            upload_timestamp=upload_ts,
            result_json=complete_pipeline_result,
            doc_type=detected_doc_type,
        )

        # 7. Phase 4 Output JSON Block
        st.subheader("📑 Phase 4 Schema Output")
        phase4_schema = {
            "doc_type": detected_doc_type,
            "extracted_fields": extracted_fields,
            "tables": extracted_tables,
        }
        with st.expander("📄 Phase 4 Extracted Fields & Tables JSON Contract", expanded=True):
            st.json(phase4_schema)
            st.download_button(
                label="📥 Download Phase 4 JSON",
                data=json.dumps(phase4_schema, indent=2),
                file_name=f"document_{doc_id[:8]}_phase4_fields.json",
                mime="application/json",
            )

        with st.expander("📄 Cumulative Complete Pipeline JSON (Phases 1-4)", expanded=False):
            st.json(complete_pipeline_result)

    except Exception as e:
        st.error(f"❌ **Processing Error:** An unexpected error occurred: {str(e)}")
        st.exception(e)
else:
    st.info("👆 Please upload a PDF, PNG, or JPG document above to begin Ingestion, Verification, OCR, and Field Extraction.")
