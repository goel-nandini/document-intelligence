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

import access_control
import authenticity_check
import chatbot
import confidence
import corrections
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


def crop_field_source_region(
    images: list[Any],
    bbox: dict[str, Any] | None,
    padding: int = 10,
) -> Any | None:
    """Crop the source image region corresponding to a field bbox with padding."""
    if not bbox or not isinstance(bbox, dict) or not images:
        return None

    page_idx = max(0, int(bbox.get("page", 1)) - 1)
    if page_idx >= len(images):
        return None

    img = images[page_idx]
    if img is None or img.size == 0:
        return None

    ih, iw = img.shape[:2]
    x = int(bbox.get("x", 0))
    y = int(bbox.get("y", 0))
    w = int(bbox.get("width", 0))
    h = int(bbox.get("height", 0))

    if w <= 0 or h <= 0 or x >= iw or y >= ih:
        return None

    # Apply padding
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(iw, x + w + padding)
    y2 = min(ih, y + h + padding)

    if x2 <= x1 or y2 <= y1:
        return None

    return img[y1:y2, x1:x2]


def determine_overall_status(
    quality_status: str,
    extracted_fields: dict[str, Any],
    cross_checks: list[dict[str, Any]],
    tamper_flags: list[dict[str, Any]],
) -> str:
    """Determine review_status.overall_status based on Phase 1-6 outcomes.

    Rules:
    - 'rejected' if quality_assessment.status was rejected_too_degraded
    - 'needs_review' if ANY field bucket is 'red' OR any cross_check passed=False OR any tamper_flag severity is 'high'
    - 'auto_accepted' otherwise.
    """
    if quality_status == "rejected_too_degraded":
        return "rejected"

    has_red_field = any(
        f.get("confidence", {}).get("bucket") == "red"
        for f in extracted_fields.values()
        if isinstance(f, dict)
    )

    has_failed_cross_check = any(
        not c.get("passed", True)
        for c in cross_checks
        if isinstance(c, dict)
    )

    has_high_tamper = any(
        str(t.get("severity", "")).lower() == "high"
        for t in tamper_flags
        if isinstance(t, dict)
    )

    if has_red_field or has_failed_cross_check or has_high_tamper:
        return "needs_review"

    return "auto_accepted"


# Sidebar with system controls & API configuration
with st.sidebar:
    st.markdown('<span class="phase-badge">Phases 1 - 9 Complete (Full Pipeline)</span>', unsafe_allow_html=True)
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
        - **Phase 3:** Exact Hash & pHash Deduplication, Timeline Inconsistency, Suspicious Software Checks, Error Level Analysis (ELA).
        - **Phase 2:** Tesseract OCR, Line Grouping, Spatial Bounding Boxes, Confidence Calibration.
        - **Phase 4:** Google Gemini Classification, Schema-Driven Field Extraction, Table Extraction & Spatial Bbox Grounding.
        - **Phase 5:** Multi-Factor Confidence Scoring (Spatial OCR Overlap, LLM Self-Reported Certainty, Format Rules).
        - **Phase 6:** Validation Layer (Cross-Checks, Font-Consistency Tamper Detection & Confidentiality Classification).
        - **Phase 7:** Review UI (On-Demand Source Region Crops, Role-Based PII Masking/Unmasking, SQLite Audit Logging).
        - **Phase 8:** Human-in-the-Loop Field Correction, Persistent Correction Memory & Few-Shot Adaptive Prompt Feedback.
        - **Phase 9:** Retrieval Query Chatbot (Confidence-Aware Citations, Masking-Compliant Fact Retrieval, Source Tracing).
        """
    )
    st.divider()
    st.markdown(f"**Blur Threshold:** `{preprocessing.BLUR_THRESHOLD}`")
    st.markdown(f"**Low Confidence OCR Cutoff:** `{ocr.LOW_CONFIDENCE_THRESHOLD * 100:.0f}%`")
    st.markdown(f"**Metadata Time Gap Threshold:** `{authenticity_check.METADATA_TIME_GAP_THRESHOLD_DAYS} day(s)`")
    st.markdown(f"**Confidence Green Cutoff:** `≥ {confidence.CONFIDENCE_GREEN_THRESHOLD:.2f}`")
    st.markdown(f"**Confidence Amber Cutoff:** `≥ {confidence.CONFIDENCE_AMBER_THRESHOLD:.2f}`")
    st.markdown(f"**Font Deviation Cutoff:** `> {validation.FONT_DEVIATION_THRESHOLD:.1f} std dev`")
    st.markdown(f"**Reviewer Demo PIN:** `{access_control.PIN_STORE['reviewer']}`")
    st.markdown(f"**Admin Demo PIN:** `{access_control.PIN_STORE['admin']}`")
    st.divider()
    if ocr.TESSERACT_AVAILABLE:
        st.success("Tesseract OCR: Online")
    else:
        st.error("Tesseract OCR: Offline (Check PATH or tesseract_cmd)")
    st.caption("Document Intelligence Engine v9.0")

# Header Section
st.markdown('<span class="phase-badge">Document Trust & Extraction Layer</span>', unsafe_allow_html=True)
st.markdown('<div class="main-header">Document Intelligence System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Automated ingestion, deduplication, authenticity check, OCR layout, schema extraction, confidence scoring, and validation layer.</div>',
    unsafe_allow_html=True,
)

def render_document_chat_tab() -> None:
    """Render Phase 9 retrieval-based chatbot UI."""
    st.markdown("### 💬 Ask About Your Documents")
    st.caption(
        "Ask natural language questions across all processed documents stored in your repository. "
        "Answers are retrieved directly from the structured trust layer (fields, tables, validation, confidence) "
        "without re-reading raw files."
    )

    all_docs = chatbot.get_all_processed_documents()

    if not all_docs:
        st.info("ℹ️ No documents processed yet — upload a document in the Ingestion tab first to start asking questions.")
    else:
        st.markdown("**Suggested Questions:**")
        q_cols = st.columns(3)
        sample_questions = [
            ("📄 Total on latest invoice?", "What's the total amount on my latest invoice?"),
            ("⚠️ Any documents flagged for review?", "Do I have any documents flagged for review?"),
            ("🔒 Any confidential documents?", "Are there any confidential documents I've uploaded?"),
        ]
        for col, (label, full_q) in zip(q_cols, sample_questions):
            with col:
                if st.button(label, key=f"chip_{label}", use_container_width=True):
                    st.session_state["pending_chat_prompt"] = full_q

    # Initialize chat history in session state
    if "doc_chat_history" not in st.session_state:
        st.session_state["doc_chat_history"] = []

    # Display existing conversation history
    for msg in st.session_state["doc_chat_history"]:
        role = msg.get("role", "user")
        with st.chat_message(role):
            st.markdown(msg.get("content", ""))
            ref_docs = msg.get("referenced_documents", [])
            if ref_docs and role == "assistant":
                for r_id in ref_docs:
                    matched_doc = next((d for d in all_docs if d.get("document_id") == r_id), None)
                    if matched_doc:
                        dtype = matched_doc.get("doc_type", "document").replace("_", " ").title()
                        dtime = matched_doc.get("upload_timestamp", "")[:10]
                        st.caption(f"📄 *Based on: {dtype} uploaded {dtime} (ID: `{r_id[:8]}`)*")
                    else:
                        st.caption(f"📄 *Based on document ID: `{r_id[:8]}`*")

    # Chat input and execution
    user_input = st.chat_input("Ask a question about your processed documents...")
    pending_prompt = st.session_state.pop("pending_chat_prompt", None)
    active_query = user_input or pending_prompt

    if active_query:
        # Display user message immediately
        st.session_state["doc_chat_history"].append({"role": "user", "content": active_query})
        with st.chat_message("user"):
            st.markdown(active_query)

        # Call chatbot backend
        with st.chat_message("assistant"):
            with st.spinner("Retrieving document intelligence facts..."):
                try:
                    result = chatbot.answer_query(
                        query=active_query,
                        chat_history=st.session_state["doc_chat_history"][:-1],
                    )
                    answer = result.get("answer", "Sorry, I couldn't process that question right now.")
                    ref_ids = result.get("referenced_documents", [])
                except Exception as err:
                    answer = "Sorry, I couldn't process that question right now."
                    ref_ids = []

            st.markdown(answer)
            if ref_ids:
                for r_id in ref_ids:
                    matched_doc = next((d for d in all_docs if d.get("document_id") == r_id), None)
                    if matched_doc:
                        dtype = matched_doc.get("doc_type", "document").replace("_", " ").title()
                        dtime = matched_doc.get("upload_timestamp", "")[:10]
                        st.caption(f"📄 *Based on: {dtype} uploaded {dtime} (ID: `{r_id[:8]}`)*")
                    else:
                        st.caption(f"📄 *Based on document ID: `{r_id[:8]}`*")

        st.session_state["doc_chat_history"].append({
            "role": "assistant",
            "content": answer,
            "referenced_documents": ref_ids,
        })
        st.rerun()


# Create Tabs: Tab 1 = Ingestion & Review, Tab 2 = Retrieval Chatbot
tab_process, tab_chat = st.tabs([
    "📄 Document Ingestion & Review",
    "💬 Ask About Your Documents",
])

with tab_process:

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

            # Maintain consistent document_id across UI interactions for this uploaded file
            file_hash = metadata["file_hash_sha256"]
            if f"active_doc_id_{file_hash}" not in st.session_state:
                st.session_state[f"active_doc_id_{file_hash}"] = doc_id
            else:
                doc_id = st.session_state[f"active_doc_id_{file_hash}"]
                prep_result["document_id"] = doc_id

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
            if duplicate_match is not None and duplicate_match["matched_document_id"] != doc_id:
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
                with st.spinner("Phase 4 & 8: Classifying document type and extracting schema fields with adaptive correction memory..."):
                    extraction_result = extraction.run_extraction_pipeline_with_memory(ocr_results)
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

            # =============================================================
            # Phase 7 & 8: Access Control, Review Status, Audit & Corrections
            # =============================================================
            masked_field_names = confidentiality.get("masked_field_names", [])
            is_confidential = confidentiality.get("is_confidential", False)

            # 0. Load any existing reviewer corrections for this document from SQLite
            try:
                with db.get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        SELECT field_name, original_value, corrected_value, corrected_by, corrected_at
                        FROM corrections
                        WHERE document_id = ?
                        ORDER BY id ASC
                        """,
                        (doc_id,),
                    )
                    session_corrections = [
                        {
                            "field_name": r["field_name"],
                            "original_value": r["original_value"] or "",
                            "corrected_value": r["corrected_value"] or "",
                            "corrected_by": r["corrected_by"] or "Reviewer",
                            "corrected_at": r["corrected_at"] or "",
                        }
                        for r in cursor.fetchall()
                    ]
            except Exception:
                session_corrections = []

            # Apply corrections to extracted_fields in session
            for corr in session_corrections:
                fn = corr["field_name"]
                if fn in extracted_fields:
                    extracted_fields[fn]["value"] = corr["corrected_value"]
                    extracted_fields[fn]["manually_corrected"] = True
                    if "confidence" in extracted_fields[fn]:
                        extracted_fields[fn]["confidence"]["bucket"] = "green"
                        extracted_fields[fn]["confidence"]["combined_score"] = 1.0

            # 1. Determine overall review status
            overall_status = determine_overall_status(
                quality_status=quality.get("status", "ok"),
                extracted_fields=extracted_fields,
                cross_checks=cross_checks,
                tamper_flags=all_tamper_flags,
            )
            review_status = {
                "overall_status": overall_status,
                "corrections": session_corrections,
            }

            # 2. Automatically log 'document_uploaded' audit event on first completion
            existing_logs = access_control.get_audit_log(doc_id)
            has_upload_event = any(log.get("action") == "document_uploaded" for log in existing_logs)
            if not has_upload_event:
                access_control.log_audit_event(
                    document_id=doc_id,
                    action="document_uploaded",
                    actor="system",
                    detail=f"Document ingested: {metadata['original_filename']} ({detected_doc_type})",
                )

            # 3. Add is_masked to each extracted_fields entry for permanent storage schema
            stored_extracted_fields = {}
            for fname, finfo in extracted_fields.items():
                f_copy = dict(finfo)
                f_copy["is_masked"] = (fname in masked_field_names)
                stored_extracted_fields[fname] = f_copy

            # 4. Fetch up-to-date audit trail
            current_audit_log = access_control.get_audit_log(doc_id)

            # 5. Update SQLite Cache with Complete Pipeline Results (Fixed Schema)
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
                "extracted_fields": stored_extracted_fields,
                "tables": extracted_tables,
                "review_status": review_status,
                "audit_log": current_audit_log,
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
            # UI DISPLAY: REVIEW UI (TABULAR & IMAGE-BASED ONLY - NO RAW JSON)
            # -------------------------------------------------------------
            st.markdown("---")
            st.markdown("## 🛡️ Document Intelligence Review Dashboard")

            # 1. REVIEW STATUS BANNER (at very top of review section)
            if overall_status == "auto_accepted":
                st.success(
                    "### ✅ Auto-Accepted\n\n"
                    "**Review Status:** All business validation cross-checks passed, zero high-severity tamper "
                    "anomalies detected, and all extracted fields met high confidence criteria."
                )
            elif overall_status == "needs_review":
                st.warning(
                    "### ⚠️ Needs Review\n\n"
                    "**Review Status:** One or more extracted fields require human verification due to low/red confidence, "
                    "failed business cross-checks, or high-severity tamper flags."
                )
            else:
                st.error(
                    "### ❌ Rejected\n\n"
                    "**Review Status:** Document failed critical quality or integrity verification thresholds."
                )

            # 2. CONFIDENTIALITY BANNER
            if is_confidential:
                sens_badge = confidentiality.get("sensitivity_level", "confidential").upper()
                reason_text = confidentiality.get("classification_reason", "Contains sensitive protected identifiers.")
                st.info(f"🔒 **Confidentiality ({sens_badge})**: {reason_text}")

            # 3. CONFIDENTIALITY VERIFY-TO-REVEAL SECTION
            # If no confidential fields exist, hide section entirely
            revealed_key = f"revealed_{doc_id}"
            is_revealed = st.session_state.get(revealed_key, False)

            if masked_field_names:
                with st.expander("🔓 Verify to Reveal Sensitive Fields", expanded=not is_revealed):
                    if not is_revealed:
                        st.markdown(
                            "This document contains protected fields subject to confidentiality masking. "
                            "Authorized reviewers or administrators can verify their security PIN to reveal sensitive values."
                        )
                        c_role, c_pin, c_btn = st.columns([2, 2, 1])
                        with c_role:
                            selected_role = st.selectbox(
                                "Role",
                                ["reviewer", "admin"],
                                key=f"role_sel_{doc_id}",
                                format_func=lambda r: "Reviewer" if r == "reviewer" else "Admin",
                            )
                        with c_pin:
                            entered_pin = st.text_input(
                                "Security PIN",
                                type="password",
                                key=f"pin_in_{doc_id}",
                                placeholder="Enter PIN",
                            )
                        with c_btn:
                            st.write("")
                            st.write("")
                            if st.button("Verify", key=f"btn_verify_pin_{doc_id}", use_container_width=True):
                                if access_control.verify_pin(selected_role, entered_pin):
                                    st.session_state[revealed_key] = True
                                    access_control.log_audit_event(
                                        document_id=doc_id,
                                        action="confidential_field_revealed",
                                        actor=selected_role,
                                        detail=f"Revealed fields: {masked_field_names}",
                                    )
                                    st.success("Verification successful! Sensitive fields revealed.")
                                    st.rerun()
                                else:
                                    access_control.log_audit_event(
                                        document_id=doc_id,
                                        action="confidential_reveal_failed",
                                        actor=selected_role,
                                        detail=f"Failed PIN verification attempt for role '{selected_role}'",
                                    )
                                    st.error("Incorrect PIN")
                    else:
                        st.success(
                            f"🔓 **Access Granted:** Sensitive fields are currently revealed for this session: "
                            f"{', '.join(f'`{f}`' for f in masked_field_names)}"
                        )
                        if st.button("🔒 Re-mask Sensitive Fields", key=f"btn_remask_{doc_id}"):
                            st.session_state[revealed_key] = False
                            st.rerun()

            # 4. Document Type Pill
            st.markdown(
                f'<div class="doc-type-pill">DOCUMENT CLASSIFICATION: {detected_doc_type.upper()}</div>',
                unsafe_allow_html=True,
            )

            # 5. Warnings for omitted / unfound fields & unmapped bounding boxes
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

            # 6. Summary Metric Row
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

            # 7. "Key Details at a Glance" (Masked vs Revealed values)
            st.markdown("---")
            st.markdown("### 🔍 Key Details at a Glance")
            formatted_doc_type = detected_doc_type.replace("_", " ").title()
            glance_lines = [f"**Document Type:** {formatted_doc_type}\n", "**Key Fields:**"]

            for fname, finfo in extracted_fields.items():
                flabel = fname.replace("_", " ").title()
                raw_v = finfo.get("value")
                is_f_masked = (fname in masked_field_names) and not is_revealed
                if is_f_masked:
                    v_str = f"🔒 {access_control.mask_value(raw_v, finfo.get('field_type', 'text'))}"
                elif (fname in masked_field_names) and is_revealed:
                    v_str = f"🔓 {raw_v}"
                else:
                    v_str = str(raw_v) if raw_v is not None else ""

                if finfo.get("manually_corrected", False):
                    badge_str = "✔️ (Manually Verified)"
                else:
                    fbucket = finfo.get("confidence", {}).get("bucket", "red")
                    badge_str = bucket_emoji_map.get(fbucket, "🔴")

                glance_lines.append(f"- **{flabel}:** {v_str} {badge_str}")

            if len(glance_lines) > 2:
                st.markdown("\n".join(glance_lines))
            else:
                st.markdown(f"**Document Type:** {formatted_doc_type}\n\n- *No key fields extracted.*")

            st.markdown("---")

            # 8. Clean st.dataframe with EXACTLY these columns:
            # Field Name | Value | Confidence | Score
            # Sorted so 🔴 red appears first, then 🟡 amber, then 🟢 green
            st.markdown("### 📋 Field Extraction & Confidence Table")
            priority_map = {"red": 0, "amber": 1, "green": 2}
            field_rows = []

            for fname, finfo in extracted_fields.items():
                flabel = fname.replace("_", " ").title()
                raw_v = finfo.get("value")
                is_f_masked = (fname in masked_field_names) and not is_revealed
                if is_f_masked:
                    v_display = f"🔒 {access_control.mask_value(raw_v, finfo.get('field_type', 'text'))}"
                elif (fname in masked_field_names) and is_revealed:
                    v_display = f"🔓 {raw_v}"
                else:
                    v_display = str(raw_v) if raw_v is not None else ""

                conf_obj = finfo.get("confidence", {})
                bucket = conf_obj.get("bucket", "red")
                score = conf_obj.get("combined_score", 0.0)

                if finfo.get("manually_corrected", False):
                    conf_badge = "✔️ Verified"
                    score_str = "1.00"
                    prio = 2
                else:
                    conf_badge = bucket_emoji_map.get(bucket, "🔴")
                    score_str = f"{score:.2f}"
                    prio = priority_map.get(bucket, 3)

                field_rows.append({
                    "Field Name": flabel,
                    "Value": v_display,
                    "Confidence": conf_badge,
                    "Score": score_str,
                    "_fname": fname,
                    "_priority": prio,
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

            # 9. SOURCE-REGION DISPLAY & HUMAN CORRECTION: On-Demand Visual Verification per Field Row
            st.markdown("#### 🔎 Source-Region Visual Verification & Field Correction")
            st.caption("Inspect source crops to verify values. Use **✏️ Correct** to adjust any incorrect extraction; edits feed directly into correction memory.")

            for r in field_rows:
                fname = r["_fname"]
                flabel = r["Field Name"]
                disp_val = r["Value"]
                f_conf_emoji = r["Confidence"]
                f_score = r["Score"]
                finfo = extracted_fields[fname]
                b = finfo.get("bbox")

                c_label, c_val, c_conf, c_score, c_btn = st.columns([3, 3, 1, 1, 2])
                with c_label:
                    st.markdown(f"**{flabel}**")
                with c_val:
                    st.markdown(f"`{disp_val}`" if disp_val else "*empty*")
                with c_conf:
                    st.markdown(f_conf_emoji)
                with c_score:
                    st.markdown(f"`{f_score}`")
                with c_btn:
                    view_key = f"view_src_{doc_id}_{fname}"
                    is_src_active = st.session_state.get(view_key, False)
                    btn_txt = "Hide Source" if is_src_active else "View Source"
                    if st.button(btn_txt, key=f"btn_src_{doc_id}_{fname}", use_container_width=True):
                        st.session_state[view_key] = not is_src_active
                        st.rerun()

                # Render cropped region on-demand right below that field's row
                if st.session_state.get(view_key, False):
                    if not b or not isinstance(b, dict):
                        st.warning("Source region unavailable")
                    else:
                        cropped = crop_field_source_region(corrected_images, b, padding=10)
                        if cropped is not None:
                            st.image(
                                cropped,
                                caption=f"Cropped Source: {flabel} (Page {b.get('page', 1)}, BBox: [{b.get('x')}, {b.get('y')}, {b.get('width')}x{b.get('height')}])",
                                use_container_width=False,
                            )
                        else:
                            st.warning("Source region unavailable")

                # Human-in-the-Loop Field Correction Expander
                with st.expander(f"✏️ Correct: {flabel}", expanded=False):
                    col_c1, col_c2 = st.columns([3, 2])
                    with col_c1:
                        corr_val_input = st.text_input(
                            f"Corrected Value for {flabel}",
                            value=str(finfo.get("value")) if finfo.get("value") is not None else "",
                            key=f"corr_val_{doc_id}_{fname}",
                        )
                    with col_c2:
                        corr_by_input = st.text_input(
                            "Corrected By",
                            value="Reviewer",
                            key=f"corr_by_{doc_id}_{fname}",
                        )

                    if st.button("Save Correction", key=f"btn_save_corr_{doc_id}_{fname}", type="primary"):
                        try:
                            corr_entry = corrections.save_correction(
                                document_id=doc_id,
                                doc_type=detected_doc_type,
                                field_name=fname,
                                original_value=finfo.get("value"),
                                corrected_value=corr_val_input,
                                corrected_by=corr_by_input,
                                ocr_context_snippet=finfo.get("raw_ocr_text", ""),
                            )
                            # Update current session
                            finfo["value"] = corr_val_input
                            finfo["manually_corrected"] = True
                            if "confidence" in finfo:
                                finfo["confidence"]["bucket"] = "green"
                                finfo["confidence"]["combined_score"] = 1.0

                            # Update session corrections and stored record
                            session_corrections.append(corr_entry)
                            stored_extracted_fields[fname]["value"] = corr_val_input
                            stored_extracted_fields[fname]["manually_corrected"] = True

                            # Recalculate status
                            new_status = determine_overall_status(
                                quality_status=quality.get("status", "ok"),
                                extracted_fields=extracted_fields,
                                cross_checks=cross_checks,
                                tamper_flags=all_tamper_flags,
                            )
                            review_status["overall_status"] = new_status
                            review_status["corrections"] = session_corrections

                            complete_pipeline_result["extracted_fields"] = stored_extracted_fields
                            complete_pipeline_result["review_status"] = review_status
                            complete_pipeline_result["audit_log"] = access_control.get_audit_log(doc_id)

                            db.save_document_record(
                                document_id=doc_id,
                                file_hash=metadata["file_hash_sha256"],
                                perceptual_hash=perceptual_hash,
                                upload_timestamp=upload_ts,
                                result_json=complete_pipeline_result,
                                doc_type=detected_doc_type,
                            )
                            st.success(f"✅ Correction saved for '{flabel}'!")
                            st.rerun()
                        except Exception as err:
                            st.error(f"❌ Failed to save correction: {str(err)}")

            # 10. Extracted Tables (st.dataframe only)
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

            # 11. Validation Layer UI (Cross-Checks, Tamper Alerts)
            st.markdown("---")
            st.subheader("🛡️ Business Cross-Checks & Authenticity Verification")

            # Validation Summary Metric Row
            total_cc = len(cross_checks)
            passed_cc = sum(1 for c in cross_checks if c.get("passed", False))
            cc_metric_val = f"{passed_cc}/{total_cc}" if total_cc > 0 else "N/A"
            tamper_flags_count = len(all_tamper_flags)

            val_col1, val_col2 = st.columns(2)
            with val_col1:
                st.metric("Cross-Checks Passed", cc_metric_val)
            with val_col2:
                st.metric("Tamper Anomaly Flags", tamper_flags_count)

            # Cross-Checks Table
            st.markdown("#### ⚖️ Business Logic Cross-Checks")
            if cross_checks:
                sorted_checks = sorted(cross_checks, key=lambda c: 0 if not c.get("passed", False) else 1)
                cc_table_rows = []
                for c in sorted_checks:
                    passed = c.get("passed", False)
                    result_str = "✅ Passed" if passed else "❌ Failed"
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

            # Tamper Flags Prominent Warning/Alert Boxes
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

            # 12. AUDIT LOG DISPLAY (Action | Actor | Timestamp | Detail)
            with st.expander("📜 Audit Log", expanded=False):
                doc_audit_logs = access_control.get_audit_log(doc_id)
                if doc_audit_logs:
                    audit_display_rows = [
                        {
                            "Action": item.get("action", ""),
                            "Actor": item.get("actor", ""),
                            "Timestamp": item.get("timestamp", ""),
                            "Detail": item.get("detail", ""),
                        }
                        for item in doc_audit_logs
                    ]
                    st.dataframe(pd.DataFrame(audit_display_rows), use_container_width=True, hide_index=True)
                else:
                    st.info("No audit log events recorded for this document.")

            # 13. EXTRACTION ACCURACY OVER TIME & ADAPTIVE LEARNING CURVE (Phase 8)
            with st.expander("📈 Extraction Accuracy Over Time & Adaptive Learning Curve", expanded=False):
                st.markdown("### Correction rate per document over time")
                st.caption(
                    "Human-in-the-loop reviewer corrections feed directly into the few-shot memory store, "
                    "driving extraction accuracy improvements on subsequent documents of this type."
                )
                stats = corrections.get_accuracy_stats(detected_doc_type)
                tot_docs = stats.get("total_documents_processed", 0)
                tot_corrs = stats.get("total_corrections_made", 0)
                corr_trend = stats.get("correction_rate_trend", [])

                c_st1, c_st2, c_st3 = st.columns(3)
                with c_st1:
                    st.metric("Total Documents Processed", tot_docs)
                with c_st2:
                    st.metric("Total Corrections Made", tot_corrs)
                with c_st3:
                    rate_val = f"{(tot_corrs / max(tot_docs, 1)):.2f}"
                    st.metric("Correction Rate / Doc", rate_val)

                if corr_trend:
                    df_trend = pd.DataFrame(corr_trend)
                    df_trend.rename(columns={"date": "Date", "correction_count": "Corrections"}, inplace=True)
                    st.line_chart(df_trend.set_index("Date"))
                else:
                    st.info("No historical correction records yet for this document classification.")

            # 14. File Download Button for structured data (Internal data download without UI JSON dumps)
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

with tab_chat:
    render_document_chat_tab()
