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
import confidence
import db
import extraction
import ocr
import preprocessing
import validation

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
    st.markdown('<span class="phase-badge">Phases 1 - 6 Active</span>', unsafe_allow_html=True)
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
            help="Used for Phase 4 extraction & Phase 5 confidence evaluation via Google Gemini.",
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
        - **Phase 5:** Multi-Factor Confidence Scoring (Spatial OCR Overlap, LLM Self-Reported Certainty, Format Rules).
        - **Phase 6:** Validation Layer (Cross-Checks, Font-Consistency Tamper Detection & Confidentiality Classification).
        """
    )
    st.divider()
    st.markdown(f"**Blur Threshold:** `{preprocessing.BLUR_THRESHOLD}`")
    st.markdown(f"**Low Confidence OCR Cutoff:** `{ocr.LOW_CONFIDENCE_THRESHOLD * 100:.0f}%`")
    st.markdown(f"**Metadata Time Gap Threshold:** `{authenticity_check.METADATA_TIME_GAP_THRESHOLD_DAYS} day(s)`")
    st.markdown(f"**Confidence Green Cutoff:** `≥ {confidence.CONFIDENCE_GREEN_THRESHOLD:.2f}`")
    st.markdown(f"**Confidence Amber Cutoff:** `≥ {confidence.CONFIDENCE_AMBER_THRESHOLD:.2f}`")
    st.markdown(f"**Font Deviation Cutoff:** `> {validation.FONT_DEVIATION_THRESHOLD:.1f} std dev`")
    st.divider()
    if ocr.TESSERACT_AVAILABLE:
        st.success("Tesseract OCR: Online")
    else:
        st.error("Tesseract OCR: Offline (Check PATH or tesseract_cmd)")
    st.caption("Document Intelligence Engine v6.0")

# Header Section
st.markdown('<span class="phase-badge">Document Trust & Extraction Layer</span>', unsafe_allow_html=True)
st.markdown('<div class="main-header">Document Intelligence System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Automated ingestion, deduplication, authenticity check, OCR layout, schema extraction, confidence scoring, and validation layer.</div>',
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

            # Phase 3 Validation summary for duplicate
            with st.expander("🛡️ Duplicate Verification Details", expanded=False):
                dup_df = pd.DataFrame([
                    {"Property": "Duplication Status", "Value": "Duplicate Found"},
                    {"Property": "Matched Document ID", "Value": str(matched_id)},
                    {"Property": "Match Method", "Value": str(match_type)},
                ])
                st.dataframe(dup_df, use_container_width=True, hide_index=True)

            # Fetch and display cached pipeline result if available
            cached_result = db.get_cached_result(matched_id)
            if cached_result:
                st.subheader("📦 Cached Pipeline Result")
                if "doc_type" in cached_result:
                    st.markdown(
                        f'<div class="doc-type-pill">CACHED DOC TYPE: {cached_result["doc_type"].upper()}</div>',
                        unsafe_allow_html=True,
                    )

                # Show cached fields if present
                if "extracted_fields" in cached_result:
                    st.markdown("**Cached Extracted Fields:**")
                    field_rows = [
                        {
                            "Field Name": k.replace("_", " ").title(),
                            "Extracted Value": v.get("value") if isinstance(v, dict) else v,
                            "Raw OCR Text": v.get("raw_ocr_text") if isinstance(v, dict) else "",
                        }
                        for k, v in cached_result["extracted_fields"].items()
                    ]
                    if field_rows:
                        st.dataframe(pd.DataFrame(field_rows), use_container_width=True, hide_index=True)

                with st.expander("📊 Normalized Cached Record Data", expanded=False):
                    try:
                        norm_df = pd.json_normalize(cached_result)
                        st.dataframe(norm_df, use_container_width=True, hide_index=True)
                    except Exception:
                        pass
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

        with st.expander("🛡️ Phase 3 Validation Audit Record", expanded=False):
            val_rows = [
                {"Audit Check": "Duplication Detected", "Result": "False (Original Document)"},
                {"Audit Check": "Tamper Anomaly Flags Count", "Result": str(len(tamper_flags))},
            ]
            st.dataframe(pd.DataFrame(val_rows), use_container_width=True, hide_index=True)

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
        # Phase 4 & 5: Schema Extraction & Per-Field Confidence Scoring
        # =============================================================
        st.divider()
        st.subheader("🧠 Phase 4 & 5: Schema Extraction & Confidence Analysis")

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

        # Part A.3: Type verification and assertions
        if not isinstance(extraction_result, dict):
            raise TypeError(
                f"Critical error: extraction_result must be a Python dict, but received {type(extraction_result).__name__}: {extraction_result}"
            )

        detected_doc_type = extraction_result.get("doc_type", "other")
        extracted_fields = extraction_result.get("extracted_fields", {})
        extracted_tables = extraction_result.get("tables", [])
        skipped_fields = extraction_result.get("_skipped_fields", [])
        unmapped_bbox_fields = extraction_result.get("_unmapped_bbox_fields", [])

        if not isinstance(extracted_fields, dict):
            raise TypeError(
                f"Critical error: extracted_fields must be a Python dict, but received {type(extracted_fields).__name__}: {extracted_fields}"
            )
        if not isinstance(extracted_tables, list):
            raise TypeError(
                f"Critical error: tables must be a Python list, but received {type(extracted_tables).__name__}: {extracted_tables}"
            )

        # Build full text for Phase 5 LLM confidence scoring
        ocr_full_text = "\n\n".join(
            f"--- Page {p.get('page_number', i + 1)} ---\n{p.get('full_text', '')}"
            for i, p in enumerate(pages_ocr)
        )

        # Phase 5: Multi-factor confidence scoring
        with st.spinner("Phase 5: Calculating spatial OCR, LLM certainty, and rule-based confidence scores..."):
            extracted_fields, extracted_tables = confidence.score_all_fields(
                extracted_fields=extracted_fields,
                tables=extracted_tables,
                ocr_results=ocr_results,
                doc_type=detected_doc_type,
                ocr_full_text=ocr_full_text,
            )

        # Double check types after confidence scoring
        assert isinstance(extracted_fields, dict), f"scored extracted_fields must be dict, got {type(extracted_fields)}"
        assert isinstance(extracted_tables, list), f"scored tables must be list, got {type(extracted_tables)}"

        # Part A.4: Print final extracted_fields and tables structure to console
        print("\n" + "=" * 60)
        print(f"[DEBUG Phase 5] FINAL SCORED EXTRACTED FIELDS (Type: {type(extracted_fields)}, Count: {len(extracted_fields)}):")
        for fn, finfo in extracted_fields.items():
            c = finfo.get("confidence", {})
            print(f"  - {fn}: val={repr(finfo.get('value'))} (val_type: {type(finfo.get('value')).__name__}), bucket={c.get('bucket')}, score={c.get('combined_score')}, ocr={c.get('ocr_confidence')}, llm={c.get('llm_confidence')}, rule={c.get('rule_check_passed')}")
        print(f"[DEBUG Phase 5] FINAL SCORED TABLES (Type: {type(extracted_tables)}, Count: {len(extracted_tables)}):")
        for tbl in extracted_tables:
            print(f"  - Table: {tbl.get('table_name')}, confidence={tbl.get('table_confidence')}, rows={len(tbl.get('rows', []))}")
        print("=" * 60 + "\n")

        # =============================================================
        # Phase 6: Validation Layer (Cross-Checks, Font Consistency & Confidentiality)
        # =============================================================
        with st.spinner("Phase 6: Running business cross-checks, font-tamper consistency & confidentiality classification..."):
            validation_output = validation.run_validation_pipeline(
                doc_type=detected_doc_type,
                extracted_fields=extracted_fields,
                tables=extracted_tables,
                ocr_results=ocr_results,
                corrected_image=corrected_images[0],
                existing_tamper_flags=tamper_flags,
            )

        cross_checks = validation_output.get("cross_checks", [])
        all_tamper_flags = validation_output.get("tamper_flags", [])
        confidentiality = validation_output.get("confidentiality", {})

        # Update SQLite Cache with Complete Pipeline Results (Phase 6 internal JSON storage)
        complete_pipeline_result = {
            "document_id": doc_id,
            "upload_timestamp": upload_ts,
            "file_metadata": metadata,
            "quality_assessment": quality,
            "validation": {
                "duplication_check": validation_schema["validation"]["duplication_check"],
                "cross_checks": cross_checks,
                "tamper_flags": all_tamper_flags,
            },
            "confidentiality": confidentiality,
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

        # -------------------------------------------------------------
        # UI DISPLAY: TABULAR AND POINTER-BASED ONLY (NO RAW JSON)
        # -------------------------------------------------------------
        # 1. Prominent Document Type Display
        st.markdown(
            f'<div class="doc-type-pill">DOCUMENT CLASSIFICATION: {detected_doc_type.upper()}</div>',
            unsafe_allow_html=True,
        )

        # 2. Warnings for omitted / unfound fields & unmapped bounding boxes
        if skipped_fields:
            st.warning(
                f"⚠️ **Omitted / Unfound Fields:** The following fields from the `{detected_doc_type}` schema "
                f"were not found in the document text: {', '.join(f'`{f}`' for f in skipped_fields)}"
            )

        if unmapped_bbox_fields:
            st.warning(
                f"⚠️ **Spatial Grounding Warning:** Bounding boxes could not be located in OCR text for: "
                f"{', '.join(f'`{f}`' for f in unmapped_bbox_fields)}"
            )

        # 3. Summary Metric Row
        bucket_emoji_map = {"green": "🟢", "amber": "🟡", "red": "🔴"}
        total_fields = len(extracted_fields)
        green_count = sum(1 for f in extracted_fields.values() if f.get("confidence", {}).get("bucket") == "green")
        amber_count = sum(1 for f in extracted_fields.values() if f.get("confidence", {}).get("bucket") == "amber")
        red_count = sum(1 for f in extracted_fields.values() if f.get("confidence", {}).get("bucket") == "red")

        st.markdown("#### 🎯 Confidence Summary")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Total Fields Extracted", total_fields)
        with m_col2:
            st.metric("🟢 Green Count", green_count)
        with m_col3:
            st.metric("🟡 Amber Count", amber_count)
        with m_col4:
            st.metric("🔴 Red Count", red_count)

        # 4. "Key Details at a Glance" (Programmatically generated bullet points)
        st.markdown("---")
        st.markdown("### 🔍 Key Details at a Glance")
        formatted_doc_type = detected_doc_type.replace("_", " ").title()
        glance_lines = [f"**Document Type:** {formatted_doc_type}\n", "**Key Fields:**"]

        for fname, finfo in extracted_fields.items():
            flabel = fname.replace("_", " ").title()
            fval = finfo.get("value", "")
            fbucket = finfo.get("confidence", {}).get("bucket", "red")
            emoji = bucket_emoji_map.get(fbucket, "🔴")
            glance_lines.append(f"- **{flabel}:** {fval} {emoji}")

        if len(glance_lines) > 2:
            st.markdown("\n".join(glance_lines))
        else:
            st.markdown(f"**Document Type:** {formatted_doc_type}\n\n- *No key fields extracted.*")

        st.markdown("---")

        # 5. Clean st.dataframe with EXACTLY these columns:
        # Field Name | Value | Confidence | Score
        # Sorted so 🔴 red appears first, then 🟡 amber, then 🟢 green
        st.markdown("### 📋 Field Extraction & Confidence Table")
        priority_map = {"red": 0, "amber": 1, "green": 2}
        field_rows = []

        for fname, finfo in extracted_fields.items():
            flabel = fname.replace("_", " ").title()
            fval = finfo.get("value")
            conf_obj = finfo.get("confidence", {})
            bucket = conf_obj.get("bucket", "red")
            score = conf_obj.get("combined_score", 0.0)
            emoji = bucket_emoji_map.get(bucket, "🔴")

            field_rows.append({
                "Field Name": flabel,
                "Value": str(fval) if fval is not None else "",
                "Confidence": emoji,
                "Score": f"{score:.2f}",
                "_priority": priority_map.get(bucket, 3),
                "_score": score,
            })

        # Sort: red first (0), amber second (1), green third (2); within bucket, lowest score first
        field_rows.sort(key=lambda r: (r["_priority"], r["_score"]))

        # Drop temporary sort keys to ensure EXACT column match: Field Name | Value | Confidence | Score
        display_field_rows = [
            {
                "Field Name": r["Field Name"],
                "Value": r["Value"],
                "Confidence": r["Confidence"],
                "Score": r["Score"],
            }
            for r in field_rows
        ]

        if display_field_rows:
            st.dataframe(pd.DataFrame(display_field_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No key-value fields were extracted from this document.")

        # 6. Extracted Tables (ONLY rendered as st.dataframe with actual columns + Confidence column)
        if extracted_tables:
            st.markdown("### 📊 Extracted Tables")
            for tbl in extracted_tables:
                t_name = tbl.get("table_name", "table")
                t_title = t_name.replace("_", " ").title()
                t_conf = tbl.get("table_confidence", 0.0)
                t_emoji = "🟢" if t_conf >= 0.75 else ("🟡" if t_conf >= 0.40 else "🔴")
                st.caption(f"**{t_title} — Table Confidence: {t_emoji} {t_conf:.2f}**")

                rows = tbl.get("rows", [])
                if rows:
                    tbl_rows_data = []
                    for r in rows:
                        row_dict = {}
                        cells = r.get("cells", {})
                        cell_confs = []
                        for col_name, cdata in cells.items():
                            c_val = cdata.get("value") if isinstance(cdata, dict) else cdata
                            c_conf = cdata.get("confidence", 0.0) if isinstance(cdata, dict) else 0.0
                            col_title = col_name.replace("_", " ").title()
                            row_dict[col_title] = str(c_val) if c_val is not None else ""
                            cell_confs.append(c_conf)

                        avg_row_conf = sum(cell_confs) / len(cell_confs) if cell_confs else 0.0
                        row_emoji = "🟢" if avg_row_conf >= 0.75 else ("🟡" if avg_row_conf >= 0.40 else "🔴")
                        row_dict["Confidence"] = row_emoji
                        tbl_rows_data.append(row_dict)

                    df_tbl = pd.DataFrame(tbl_rows_data)
                    st.dataframe(df_tbl, use_container_width=True, hide_index=True)
                else:
                    st.caption(f"Table '{t_title}' was detected but contains no data rows.")

        # =============================================================
        # Phase 6: Validation Layer UI (Cross-Checks, Tamper Alerts, Confidentiality)
        # =============================================================
        st.markdown("---")
        st.subheader("🛡️ Phase 6: Validation & Authenticity Layer")

        # 1. Validation Summary Metric Row
        total_cc = len(cross_checks)
        passed_cc = sum(1 for c in cross_checks if c.get("passed", False))
        cc_metric_val = f"{passed_cc}/{total_cc}" if total_cc > 0 else "N/A"

        tamper_flags_count = len(all_tamper_flags)

        is_confidential = confidentiality.get("is_confidential", False)
        sensitivity_level = confidentiality.get("sensitivity_level", "none").upper()
        if is_confidential:
            conf_metric_val = f"🔒 {sensitivity_level}"
        else:
            conf_metric_val = "🔓 UNRESTRICTED"

        val_col1, val_col2, val_col3 = st.columns(3)
        with val_col1:
            st.metric("Cross-Checks Passed", cc_metric_val)
        with val_col2:
            st.metric("Tamper Anomaly Flags", tamper_flags_count)
        with val_col3:
            st.metric("Confidentiality Status", conf_metric_val)

        # 2. Business Logic Cross-Checks Table
        st.markdown("#### ⚖️ Business Logic Cross-Checks")
        if cross_checks:
            # Sort failed checks first
            sorted_checks = sorted(cross_checks, key=lambda c: 0 if not c.get("passed", False) else 1)
            cc_table_rows = []
            for c in sorted_checks:
                passed = c.get("passed", False)
                result_str = "✅ Passed" if passed else "❌ Failed"
                fields_cmp = ", ".join(f"`{f}`" for f in c.get("fields_compared", []))
                detail = c.get("discrepancy_detail") or c.get("expected_relationship", "OK")

                cc_table_rows.append({
                    "Check Name": c.get("check_name", "").replace("_", " ").title(),
                    "Fields Compared": ", ".join(c.get("fields_compared", [])),
                    "Result": result_str,
                    "Detail": detail,
                })
            st.dataframe(pd.DataFrame(cc_table_rows), use_container_width=True, hide_index=True)
        else:
            st.info(f"No specific business cross-checks configured for '{detected_doc_type}' documents.")

        # 3. Tamper Flags Prominent Warning/Alert Boxes (Immediately visible, never hidden)
        st.markdown("#### 🚩 Tamper & Authenticity Alerts")
        if all_tamper_flags:
            for flag in all_tamper_flags:
                f_type = flag.get("flag_type", "tamper_flag").replace("_", " ").upper()
                f_field = flag.get("affected_field")
                f_target = f"Field: `{f_field}`" if f_field else "Whole Document"
                f_sev = flag.get("severity", "medium").upper()
                f_evidence = flag.get("evidence", "")

                alert_text = (
                    f"**[{f_type}] — {f_sev} Severity** | **Target:** {f_target}\n\n"
                    f"{f_evidence}"
                )
                if f_sev == "HIGH":
                    st.error(f"🚨 {alert_text}")
                elif f_sev == "MEDIUM":
                    st.warning(f"⚠️ {alert_text}")
                else:
                    st.info(f"ℹ️ {alert_text}")
        else:
            st.success(
                "✅ **Authenticity Verification Passed:** No metadata inconsistencies, "
                "recompression artifacts, or font irregularities detected across the document."
            )

        # 4. Confidentiality & PII Classification Highlight
        st.markdown("#### 🔒 Confidentiality & PII Classification")
        if is_confidential:
            masked_fields = confidentiality.get("masked_field_names", [])
            masked_badges = ", ".join(f"`{f}`" for f in masked_fields) if masked_fields else "None"
            reason = confidentiality.get("classification_reason", "")

            st.info(
                f"🔒 **This document contains sensitive information ({sensitivity_level} SENSITIVITY).**\n\n"
                f"**Classification Reason:** {reason}\n\n"
                f"**Default Masked Fields:** {masked_badges}"
            )
        else:
            st.success(
                "🔓 **Standard Unrestricted Document:** No sensitive personally identifiable information (PII) "
                "or protected financial identifiers were detected in this document."
            )

        # 7. Flattened Audit View using pd.json_normalize() (NO raw JSON)
        with st.expander("🔍 Flattened Audit Log & Spatial Coordinates", expanded=False):
            audit_records = []
            for fname, finfo in extracted_fields.items():
                c = finfo.get("confidence", {})
                b = finfo.get("bbox")
                bbox_str = f"Page {b.get('page', 1)} [{b.get('x')}, {b.get('y')}, {b.get('width')}x{b.get('height')}]" if b else "Unmapped"
                audit_records.append({
                    "Field Name": fname,
                    "Value": finfo.get("value"),
                    "Raw OCR Text": finfo.get("raw_ocr_text"),
                    "Field Type": finfo.get("field_type"),
                    "OCR Confidence": c.get("ocr_confidence"),
                    "LLM Confidence": c.get("llm_confidence"),
                    "Rule Passed": c.get("rule_check_passed"),
                    "Combined Score": c.get("combined_score"),
                    "Bucket": c.get("bucket"),
                    "Source Bounding Box": bbox_str,
                })
            if audit_records:
                st.dataframe(pd.json_normalize(audit_records), use_container_width=True, hide_index=True)

        # 8. File Download Button for structured data (Internal data download without UI JSON dumps)
        st.download_button(
            label="📥 Download Extracted Intelligence Record (JSON)",
            data=json.dumps(complete_pipeline_result, indent=2),
            file_name=f"document_{doc_id[:8]}_intelligence_record.json",
            mime="application/json",
        )

    except Exception as e:
        st.error(f"❌ **Processing Error:** An unexpected error occurred: {str(e)}")
        st.exception(e)
else:
    st.info("👆 Please upload a PDF, PNG, or JPG document above to begin Ingestion, Verification, OCR, and Field Extraction.")
