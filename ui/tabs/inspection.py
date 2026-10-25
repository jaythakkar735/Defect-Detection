"""
Tab 1: Single-Item Quality Inspection (Industrial Control-Room).
Implements a 3-state decision engine, horizontal calibrated score gauge,
ground-truth reconciliation, and session inspection logging with CSV export.
"""
from datetime import datetime
import streamlit as st
import pandas as pd
from PIL import Image
from config import get_decision_state, get_reliability_tier, REVIEW_BAND_MARGIN
from ui.components import render_score_gauge, render_decision_card
from ui.terms import get_term_label, get_term_help

def render_inspection_tab(detector, eval_res, test_ds, transform, selected_category: str, audience_mode: str):
    st.markdown("### QUALITY INSPECTION STATION // INLINE QA")
    
    tier_info = get_reliability_tier(eval_res["auroc"] * 100.0)
    
    # How to read this page expander
    with st.expander("HOW TO OPERATE THIS STATION // PROTOCOL"):
        st.markdown(f"""
        - **Target Domain:** Category **{selected_category.upper()}** ({tier_info['badge']}).
        - **Decision Policy:** 
          - **PASS:** Item anomaly score is below `{eval_res['threshold'] * (1.0 - REVIEW_BAND_MARGIN):.2f}`.
          - **REVIEW BAND (Buffer):** Scores between `{eval_res['threshold'] * (1.0 - REVIEW_BAND_MARGIN):.2f}` and `{eval_res['threshold']:.2f}` require human inspection.
          - **REJECT:** Scores $\\ge$ `{eval_res['threshold']:.2f}` exceed the calibrated nominal boundary.
        - **Display-only Note:** The review band is a visual triage aid; underlying binary scoring is strictly preserved.
        """)

    col_select, col_upload = st.columns([1, 1])
    
    with col_select:
        sample_options = [
            f"Sample {i+1:02d} [{item['defect_type'].upper()}] - {item.get('filename', item['image_path'].name)}"
            for i, item in enumerate(test_ds.samples)
        ]
        selected_idx = st.selectbox(
            "Benchmark Test Library Sample:",
            range(len(sample_options)),
            format_func=lambda i: sample_options[i],
            help="Select an item from the verified benchmark test set containing both nominal and defective samples."
        )
        selected_item = test_ds.samples[selected_idx]

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload Ad-Hoc Production Image (PNG/JPG):",
            type=["png", "jpg", "jpeg"],
            help="Upload an ad-hoc manufacturing image to run through the feature extraction engine."
        )

    # Resolve image source and ground truth
    if uploaded_file is not None:
        inspect_image = Image.open(uploaded_file).convert("RGB")
        img_title = f"Uploaded Custom Image: {uploaded_file.name}"
        ground_truth_mask = None
        true_label_str = "Unknown"
        item_identifier = uploaded_file.name
    else:
        inspect_image = Image.open(selected_item["image_path"]).convert("RGB")
        img_title = f"Item: {selected_item.get('filename', selected_item['image_path'].name)} | Subtype: {selected_item['defect_type']}"
        ground_truth_mask = Image.open(selected_item["mask_path"]) if selected_item.get("mask_path") else None
        true_label_str = "Defective" if selected_item["label"] == 1 else "Normal"
        item_identifier = selected_item.get("filename", selected_item["image_path"].name)

    # Compute inference score and 3-state decision
    score = detector.score_single_image(inspect_image, transform)
    threshold = float(eval_res["threshold"])
    decision = get_decision_state(score, threshold, margin_ratio=REVIEW_BAND_MARGIN)

    # Session log tracking (Prevent duplicate logging on rerun)
    if "inspection_log" not in st.session_state:
        st.session_state["inspection_log"] = []

    last_logged = st.session_state.get("last_logged_item", None)
    current_log_signature = (selected_category, item_identifier, round(score, 4))
    
    if last_logged != current_log_signature:
        st.session_state["inspection_log"].append({
            "Timestamp": datetime.now().strftime("%H:%M:%S"),
            "Category": selected_category.capitalize(),
            "Item": item_identifier,
            "Score": round(score, 3),
            "Limit": round(threshold, 3),
            "Decision": decision["label"],
            "Ground Truth": true_label_str
        })
        st.session_state["last_logged_item"] = current_log_signature

    st.markdown("---")

    # Image display & Decision Layout
    col_img, col_verdict = st.columns([1, 1], gap="medium")

    with col_img:
        st.image(inspect_image, caption=img_title, use_container_width=True)
        
        # Heatmap hook (clean toggle without fabricating)
        show_heatmap = st.checkbox("Show feature activation localization overlay", value=False, help="Localization overlay hook. When pixel-level backpropagation or PatchCore feature banks are active, heatmaps render here.")
        if show_heatmap:
            if ground_truth_mask is not None:
                st.image(ground_truth_mask, caption="Ground-Truth Annotation Mask (Defect Footprint)", use_container_width=True)
            else:
                st.info("Feature localization hook active: global pooled backbone evaluates full-image context.")

    with col_verdict:
        # Decision Card with false negative/positive honesty
        render_decision_card(decision, score, threshold, true_label_str, audience_mode)
        
        # Horizontal Score Gauge
        st.markdown(f"**{get_term_label('anomaly_score', audience_mode).upper()} CALIBRATION GAUGE**")
        render_score_gauge(score, threshold, decision["review_limit"])

        # Metric Tiles
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(
                label=get_term_label("anomaly_score", audience_mode),
                value=f"{score:.2f}",
                help=get_term_help("anomaly_score")
            )
        with col_m2:
            st.metric(
                label=get_term_label("threshold", audience_mode),
                value=f"{threshold:.2f}",
                help=get_term_help("threshold")
            )

    # Session Inspection History Log
    st.markdown("---")
    st.markdown("#### STATION EVENT LOG // RECENT INSPECTIONS")
    
    if st.session_state["inspection_log"]:
        log_df = pd.DataFrame(st.session_state["inspection_log"])
        st.dataframe(log_df, use_container_width=True, hide_index=True)
        
        csv_data = log_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="EXPORT SESSION LOG (CSV)",
            data=csv_data,
            file_name=f"inspection_log_{selected_category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="Download all decisions made during this active browser session."
        )
