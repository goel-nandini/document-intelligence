"""Document Intelligence System.

Streamlit application providing automated document ingestion, deduplication,
authenticity analysis, OCR layout extraction, schema-driven field extraction,
confidence scoring, validation cross-checks, audit trails, and interactive retrieval chatbot.
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
    initial_sidebar_state="collapsed",
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
    
    [data-testid="stSidebar"], section[data-testid="stSidebar"] {
        display: none !important;
    }
    
    .header-badge {
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
    """Determine review_status.overall_status based on verification outcomes.

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



# Header Section
st.markdown('<span class="header-badge">Enterprise Document Intelligence Engine</span>', unsafe_allow_html=True)
st.markdown('<div class="main-header">Document Intelligence System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Automated ingestion, deduplication, authenticity check, OCR layout, schema extraction, confidence scoring, and validation layer.</div>',
    unsafe_allow_html=True,
)

def render_document_chat_tab() -> None:
    """Render retrieval-based chatbot UI."""
    st.markdown("### 💬 Ask About Your Documents")
    st.caption(
        "Ask natural language questions across all processed documents stored in your repository. "
        "Answers are retrieved directly from the structured trust layer (fields, tables, validation, confidence) "
        "without re-reading raw files."
    )

    all_docs = chatbot.get_all_processed_documents()

    # Knowledge Base Status & Quick Controls
    col_stat, col_clear = st.columns([4, 1])
    with col_stat:
        if all_docs:
            doc_badges = []
            for d in all_docs:
                fname = d.get("file_metadata", {}).get("original_filename") or d.get("doc_type", "document")
                dtype = d.get("doc_type", "doc").upper()
                doc_badges.append(f"`{fname}` ({dtype})")
            st.markdown(f"📚 **Knowledge Base Active:** {len(all_docs)} document(s) indexed: {', '.join(doc_badges)}")
        else:
            st.info("ℹ️ **No documents processed yet.** Please upload a document in the **Document Ingestion & Review** tab first to start asking questions.")

    with col_clear:
        if st.session_state.get("doc_chat_history"):
            if st.button("🗑️ Clear Chat", key="btn_clear_chat", use_container_width=True):
                st.session_state["doc_chat_history"] = []
                st.rerun()

    if all_docs:
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
                    st.rerun()

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
                        fname = matched_doc.get("file_metadata", {}).get("original_filename")
                        dtype = matched_doc.get("doc_type", "document").replace("_", " ").title()
                        dtime = matched_doc.get("upload_timestamp", "")[:10]
                        label_name = f"`{fname}` ({dtype})" if fname else dtype
                        st.caption(f"📄 *Based on: {label_name} uploaded {dtime} (ID: `{r_id[:8]}`)*")
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
                        fname = matched_doc.get("file_metadata", {}).get("original_filename")
                        dtype = matched_doc.get("doc_type", "document").replace("_", " ").title()
                        dtime = matched_doc.get("upload_timestamp", "")[:10]
                        label_name = f"`{fname}` ({dtype})" if fname else dtype
                        st.caption(f"📄 *Based on: {label_name} uploaded {dtime} (ID: `{r_id[:8]}`)*")
                    else:
                        st.caption(f"📄 *Based on document ID: `{r_id[:8]}`*")

        st.session_state["doc_chat_history"].append({
            "role": "assistant",
            "content": answer,
            "referenced_documents": ref_ids,
        })


def render_ingestion_review_tab() -> None:
    """Render Ingestion, Quality, Authenticity, OCR, Extraction, Validation & Review UI with clean, modern executive presentation."""
    # File Uploader
    uploaded_file = st.file_uploader(
        "Choose a document to analyze",
        type=["pdf", "png", "jpg", "jpeg"],
        help="Supported formats: PDF, PNG, JPG, JPEG",
    )

    if uploaded_file is None:
        st.info("👆 Please upload a PDF, PNG, or JPG document above to automatically extract data and verify authenticity.")
        return

    try:
        # -------------------------------------------------------------
        # Backend Processing Pipeline (Seamless execution)
        # -------------------------------------------------------------
        with st.spinner("Analyzing document structure, extracting text & reading key information..."):
            prep_result = preprocessing.preprocess_document(uploaded_file)

        doc_id = prep_result["document_id"]
        upload_ts = prep_result["upload_timestamp"]
        metadata = prep_result["file_metadata"]
        quality = prep_result["quality_assessment"]
        original_images = prep_result["original_images"]
        corrected_images = prep_result["corrected_images"]
        page_assessments = prep_result.get("page_assessments", [])

        # Maintain consistent document_id across UI interactions
        file_hash = metadata["file_hash_sha256"]
        if f"active_doc_id_{file_hash}" not in st.session_state:
            st.session_state[f"active_doc_id_{file_hash}"] = doc_id
        else:
            doc_id = st.session_state[f"active_doc_id_{file_hash}"]
            prep_result["document_id"] = doc_id

        # Quality Gate Check
        if quality.get("status") == "rejected_too_degraded":
            st.error(
                f"🛑 **Document Unreadable / Too Degraded:** The uploaded document has excessive blur "
                f"(quality score: {quality['blur_score']:.1f}). Downstream processing stopped."
            )
            return

        # Authenticity & Duplicate Check
        perceptual_hash = authenticity_check.compute_perceptual_hash(corrected_images[0])
        duplicate_match = db.find_duplicate(
            file_hash=metadata["file_hash_sha256"],
            perceptual_hash=perceptual_hash,
            similarity_threshold=5,
        )

        if duplicate_match is not None and duplicate_match["matched_document_id"] != doc_id:
            matched_id = duplicate_match["matched_document_id"]
            match_type = duplicate_match["match_type"]
            st.warning(
                f"⚡ **Previously Processed Document Found (Match: {match_type}):** "
                f"This document is already indexed in your repository (ID: `{matched_id[:8]}`)."
            )
            cached_result = db.get_cached_result(matched_id)
            if cached_result and "extracted_fields" in cached_result:
                st.markdown("### 📋 Stored Extracted Data")
                c_fields = cached_result["extracted_fields"]
                c_rows = [
                    {"Field": k.replace("_", " ").title(), "Value": str(v.get("value") if isinstance(v, dict) else v)}
                    for k, v in c_fields.items()
                ]
                st.dataframe(pd.DataFrame(c_rows), use_container_width=True, hide_index=True)
            return

        # Authenticity Pre-check & OCR
        tamper_flags = authenticity_check.run_authenticity_precheck(
            image=corrected_images[0],
            file_metadata=metadata,
        )

        ocr_output = ocr.run_ocr_on_document(corrected_images)
        ocr_results = ocr_output["ocr_results"]
        pages_ocr = ocr_results.get("pages", [])

        # Schema Extraction with Adaptive Correction Memory
        extraction_result = extraction.run_extraction_pipeline_with_memory(ocr_results)
        if not isinstance(extraction_result, dict):
            st.error("Extraction error: unexpected response format.")
            return

        detected_doc_type = extraction_result.get("doc_type", "other")
        extracted_fields = extraction_result.get("extracted_fields", {})
        extracted_tables = extraction_result.get("tables", [])
        skipped_fields = extraction_result.get("_skipped_fields", [])
        unmapped_bbox_fields = extraction_result.get("_unmapped_bbox_fields", [])

        # Confidence Scoring
        ocr_full_text = "\n\n".join(
            f"--- Page {p.get('page_number', i + 1)} ---\n{p.get('full_text', '')}"
            for i, p in enumerate(pages_ocr)
        )
        extracted_fields, extracted_tables = confidence.score_all_fields(
            extracted_fields=extracted_fields,
            tables=extracted_tables,
            ocr_results=ocr_results,
            doc_type=detected_doc_type,
            ocr_full_text=ocr_full_text,
        )

        # Validation Checks & Confidentiality
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
        masked_field_names = confidentiality.get("masked_field_names", [])
        is_confidential = confidentiality.get("is_confidential", False)

        # Load Existing Reviewer Corrections
        try:
            with db.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT field_name, original_value, corrected_value, corrected_by, corrected_at FROM corrections WHERE document_id = ? ORDER BY id ASC",
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

        for corr in session_corrections:
            fn = corr["field_name"]
            if fn in extracted_fields:
                extracted_fields[fn]["value"] = corr["corrected_value"]
                extracted_fields[fn]["manually_corrected"] = True
                if "confidence" in extracted_fields[fn]:
                    extracted_fields[fn]["confidence"]["bucket"] = "green"
                    extracted_fields[fn]["confidence"]["combined_score"] = 1.0

        # Review Status
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

        # Auto Audit Event on upload
        existing_logs = access_control.get_audit_log(doc_id)
        if not any(log.get("action") == "document_uploaded" for log in existing_logs):
            access_control.log_audit_event(
                document_id=doc_id,
                action="document_uploaded",
                actor="system",
                detail=f"Document ingested: {metadata['original_filename']} ({detected_doc_type})",
            )

        # Save to SQLite
        stored_extracted_fields = {}
        for fname, finfo in extracted_fields.items():
            f_copy = dict(finfo)
            f_copy["is_masked"] = (fname in masked_field_names)
            stored_extracted_fields[fname] = f_copy

        current_audit_log = access_control.get_audit_log(doc_id)
        complete_pipeline_result = {
            "document_id": doc_id,
            "upload_timestamp": upload_ts,
            "file_metadata": metadata,
            "quality_assessment": quality,
            "validation": {
                "duplication_check": {"is_duplicate": False, "matched_document_id": None, "match_type": None},
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
        # CLEAN & BEAUTIFUL USER INTERFACE (Simplicity & Clarity First)
        # -------------------------------------------------------------
        st.markdown("---")

        # 1. Executive Summary Card
        formatted_type = detected_doc_type.replace("_", " ").title()
        col_hdr_left, col_hdr_right = st.columns([3, 1])
        with col_hdr_left:
            st.markdown(f"## 📄 {metadata['original_filename']}")
            st.caption(f"**Classification:** `{formatted_type}` | **Pages:** {metadata['page_count']} | **Analyzed:** {upload_ts[:10]}")
        with col_hdr_right:
            st.write("")
            if overall_status == "auto_accepted":
                st.success("✅ **Verified & Accepted**")
            elif overall_status == "needs_review":
                st.warning("⚠️ **Review Recommended**")
            else:
                st.error("❌ **Action Required**")

        # Summary Metrics in Clean Simple Language
        bucket_emoji_map = {"green": "🟢", "amber": "🟡", "red": "🔴"}
        total_fields = len(extracted_fields)
        green_count = sum(1 for f in extracted_fields.values() if f.get("confidence", {}).get("bucket") == "green")
        accuracy_pct = int((green_count / max(total_fields, 1)) * 100)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Information Fields", f"{total_fields} extracted")
        with m2:
            st.metric("Confidence Level", f"{accuracy_pct}% High")
        with m3:
            failed_cc = sum(1 for c in cross_checks if not c.get("passed", False))
            st.metric("Calculation Checks", "All Passed" if failed_cc == 0 else f"{failed_cc} Mismatch")
        with m4:
            st.metric("Authenticity", "Verified Original" if not all_tamper_flags else f"{len(all_tamper_flags)} Warning(s)")

        # 2. Confidentiality PIN Unlock (Only shown if document contains masked fields)
        revealed_key = f"revealed_{doc_id}"
        is_revealed = st.session_state.get(revealed_key, False)

        if masked_field_names:
            if not is_revealed:
                with st.expander("🔒 Protected Fields: Click to Unlock with PIN", expanded=False):
                    st.caption("This document contains sensitive personal information. Enter your security PIN to view unmasked values.")
                    c_role, c_pin, c_btn = st.columns([2, 2, 1])
                    with c_role:
                        selected_role = st.selectbox("Role", ["reviewer", "admin"], key=f"role_sel_{doc_id}", format_func=lambda r: "Reviewer" if r == "reviewer" else "Admin")
                    with c_pin:
                        entered_pin = st.text_input("PIN", type="password", key=f"pin_in_{doc_id}", placeholder="Enter PIN")
                    with c_btn:
                        st.write("")
                        st.write("")
                        if st.button("Unlock", key=f"btn_verify_pin_{doc_id}", use_container_width=True):
                            if access_control.verify_pin(selected_role, entered_pin):
                                st.session_state[revealed_key] = True
                                access_control.log_audit_event(doc_id, "confidential_field_revealed", selected_role, f"Revealed: {masked_field_names}")
                                st.success("Fields unlocked!")
                                st.rerun()
                            else:
                                access_control.log_audit_event(doc_id, "confidential_reveal_failed", selected_role, "Incorrect PIN")
                                st.error("Incorrect PIN")
            else:
                st.info(f"🔓 **Access Granted:** Sensitive fields are currently visible.")
                if st.button("🔒 Re-mask Sensitive Fields", key=f"btn_remask_{doc_id}"):
                    st.session_state[revealed_key] = False
                    st.rerun()

        # 3. Main Extracted Information Section
        st.markdown("### 📋 Extracted Information")
        st.caption("Here is the structured data extracted from your document. You can inspect the source crop or make corrections directly.")

        # Prepare field rows sorted by priority
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
                prio = 0 if bucket == "red" else (1 if bucket == "amber" else 2)

            field_rows.append({
                "Field Name": flabel,
                "Value": v_display,
                "Confidence": conf_badge,
                "Score": score_str,
                "_fname": fname,
                "_priority": prio,
                "_score": score,
            })

        # Sort so any lower-confidence fields appear at top for easy review
        field_rows.sort(key=lambda r: (r["_priority"], r["_score"]))

        # Render Unified Clean Table & Actions
        for r in field_rows:
            fname = r["_fname"]
            flabel = r["Field Name"]
            disp_val = r["Value"]
            f_conf_emoji = r["Confidence"]
            f_score = r["Score"]
            finfo = extracted_fields[fname]
            b = finfo.get("bbox")

            c_label, c_val, c_conf, c_btn = st.columns([3, 4, 2, 2])
            with c_label:
                st.markdown(f"**{flabel}**")
            with c_val:
                st.markdown(f"`{disp_val}`" if disp_val else "*not specified*")
            with c_conf:
                st.markdown(f"{f_conf_emoji} `{f_score}`")
            with c_btn:
                view_key = f"view_src_{doc_id}_{fname}"
                is_src_active = st.session_state.get(view_key, False)
                btn_txt = "Hide Crop" if is_src_active else "View Crop"
                if st.button(btn_txt, key=f"btn_src_{doc_id}_{fname}", use_container_width=True):
                    st.session_state[view_key] = not is_src_active
                    st.rerun()

            # Render cropped image on demand right beneath the field
            if st.session_state.get(view_key, False):
                if b and isinstance(b, dict):
                    cropped = crop_field_source_region(corrected_images, b, padding=10)
                    if cropped is not None:
                        st.image(cropped, caption=f"Source Image Crop: {flabel}", use_container_width=False)
                    else:
                        st.caption("*(Source crop unavailable for this field)*")
                else:
                    st.caption("*(Source bounding box not located)*")

            # Inline Edit option
            with st.expander(f"✏️ Correct value for {flabel}", expanded=False):
                col_c1, col_c2 = st.columns([3, 2])
                with col_c1:
                    corr_val_input = st.text_input(f"New Value", value=str(finfo.get("value")) if finfo.get("value") is not None else "", key=f"corr_val_{doc_id}_{fname}")
                with col_c2:
                    corr_by_input = st.text_input("Reviewer Name", value="Reviewer", key=f"corr_by_{doc_id}_{fname}")

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
                        finfo["value"] = corr_val_input
                        finfo["manually_corrected"] = True
                        if "confidence" in finfo:
                            finfo["confidence"]["bucket"] = "green"
                            finfo["confidence"]["combined_score"] = 1.0

                        session_corrections.append(corr_entry)
                        stored_extracted_fields[fname]["value"] = corr_val_input
                        stored_extracted_fields[fname]["manually_corrected"] = True
                        review_status["overall_status"] = determine_overall_status(
                            quality_status=quality.get("status", "ok"),
                            extracted_fields=extracted_fields,
                            cross_checks=cross_checks,
                            tamper_flags=all_tamper_flags,
                        )
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
                        st.success(f"Correction saved for '{flabel}'!")
                        st.rerun()
                    except Exception as err:
                        st.error(f"Failed to save correction: {str(err)}")

        # 4. Extracted Tables (if present)
        if extracted_tables:
            st.markdown("### 📊 Extracted Tables")
            for tbl in extracted_tables:
                t_name = tbl.get("table_name", "Items").replace("_", " ").title()
                rows = tbl.get("rows", [])
                if rows:
                    tbl_rows_data = []
                    for r in rows:
                        row_dict = {}
                        cells = r.get("cells", {})
                        for col_name, cdata in cells.items():
                            c_val = cdata.get("value") if isinstance(cdata, dict) else cdata
                            row_dict[col_name.replace("_", " ").title()] = str(c_val) if c_val is not None else ""
                        tbl_rows_data.append(row_dict)
                    st.caption(f"**{t_name} ({len(rows)} items)**")
                    st.dataframe(pd.DataFrame(tbl_rows_data), use_container_width=True, hide_index=True)

        # 5. Document Trust & Verification Summary (Clean 3-column overview)
        st.markdown("---")
        st.markdown("### 🛡️ Trust & Integrity Checks")
        chk_col1, chk_col2, chk_col3 = st.columns(3)

        with chk_col1:
            st.markdown("**Authenticity Verification**")
            if all_tamper_flags:
                for flag in all_tamper_flags:
                    st.warning(f"⚠️ {flag.get('evidence', 'Potential anomaly detected')}")
            else:
                st.success("✅ Document is genuine (No tampering or irregularities detected).")

        with chk_col2:
            st.markdown("**Calculation Consistency**")
            if cross_checks:
                failed_checks = [c for c in cross_checks if not c.get("passed", False)]
                if failed_checks:
                    for fc in failed_checks:
                        st.warning(f"⚠️ {fc.get('check_name', '').replace('_', ' ').title()}: {fc.get('discrepancy_detail', 'Mismatch')}")
                else:
                    st.success("✅ All math calculations match correctly.")
            else:
                st.info("No math checks required for this document type.")

        with chk_col3:
            st.markdown("**Privacy & Confidentiality**")
            if is_confidential:
                st.info(f"🔒 Classified as confidential ({confidentiality.get('sensitivity_level', '').upper()}).")
            else:
                st.success("✅ Standard document (No sensitive PII flags).")

        # 6. Technical Details & Audit Logs (Tucked neatly into collapsed expanders at bottom)
        st.markdown("---")
        st.markdown("#### ⚙️ Technical Details & Tools")

        with st.expander("🖼️ Document Quality & Image Preprocessing", expanded=False):
            st.caption(f"Blur Score: `{quality['blur_score']:.1f}` | Skew Angle: `{quality['skew_angle_degrees']:.2f}°` | Status: `{quality['status']}`")
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.markdown("**Original Document**")
                st.image(original_images[0], use_container_width=True)
            with col_img2:
                st.markdown("**Deskewed & Denoised Document**")
                st.image(corrected_images[0], use_container_width=True)

        with st.expander("📄 Optical Character Recognition (OCR) Text & Word Layout", expanded=False):
            if pages_ocr:
                p1_words = pages_ocr[0].get("words", [])
                p1_text = pages_ocr[0].get("full_text", "")
                st.caption(f"Recognized {len(p1_words)} words with average confidence {ocr_results.get('average_confidence', 0.0) * 100:.1f}%.")
                col_o1, col_o2 = st.columns(2)
                with col_o1:
                    annotated_img = ocr.draw_ocr_bounding_boxes(corrected_images[0], p1_words)
                    st.image(annotated_img, caption="Spatial Bounding Box Overlay", use_container_width=True)
                with col_o2:
                    st.text_area("Extracted Plain Text", value=p1_text, height=350, key="ocr_full_preview")

        with st.expander("📜 Audit Trail & Security Logs", expanded=False):
            doc_audit_logs = access_control.get_audit_log(doc_id)
            if doc_audit_logs:
                st.dataframe(pd.DataFrame(doc_audit_logs)[["action", "actor", "timestamp", "detail"]], use_container_width=True, hide_index=True)
            else:
                st.info("No audit events recorded.")

        with st.expander("📈 Continuous Learning & Accuracy Curve", expanded=False):
            stats = corrections.get_accuracy_stats(detected_doc_type)
            trend = stats.get("correction_rate_trend", [])
            st.caption(f"Total documents processed: {stats.get('total_documents_processed', 0)} | Corrections: {stats.get('total_corrections_made', 0)}")
            if trend:
                df_trend = pd.DataFrame(trend).rename(columns={"date": "Date", "correction_count": "Corrections"})
                st.line_chart(df_trend.set_index("Date"))
            else:
                st.info("No correction history recorded yet.")

        # Download Extracted Record JSON
        st.download_button(
            label="📥 Download Extracted Record (JSON)",
            data=json.dumps(complete_pipeline_result, indent=2),
            file_name=f"{metadata['original_filename']}_extracted_record.json",
            mime="application/json",
        )

    except Exception as e:
        st.error(f"❌ An error occurred while analyzing the document: {str(e)}")
        st.exception(e)


# =====================================================================
# Main Application Tabs: Ingestion/Review vs Query Chatbot
# =====================================================================
tab_process, tab_chat = st.tabs([
    "📄 Document Ingestion & Review",
    "💬 Ask About Your Documents",
])

with tab_process:
    render_ingestion_review_tab()

with tab_chat:
    render_document_chat_tab()
