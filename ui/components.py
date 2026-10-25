"""
Reusable UI Components for the Industrial Control-Room Interface.
Includes KPI cards, status badges, score gauge, header banner, and session log.
"""
import streamlit as st
import pandas as pd
from config import get_reliability_tier, get_decision_state, REVIEW_BAND_MARGIN

def render_header_bar(category: str, mode: str, device: str, auroc: float, net_savings: float = 0.0):
    """Renders the top industrial control status bar."""
    tier_info = get_reliability_tier(auroc, net_savings)
    
    status_color = "#2BD4A0" if tier_info["status"] == "pass" else ("#F5A623" if tier_info["status"] == "info" or tier_info["status"] == "review" else "#FF5C5C")
    
    html = (
        '<div class="scada-header-bar">'
        '<div>'
        f'<span class="scada-header-title">INSPECTION CONTROL // {category.upper()}</span>'
        f'<span class="scada-chip" style="margin-left: 12px; color: {status_color}; border-color: {status_color};">'
        f'{tier_info["badge"]}'
        '</span>'
        '</div>'
        '<div class="scada-header-meta">'
        f'<span>DEVICE: <strong style="color: #E6EAF0;">{device}</strong></span>'
        f'<span>AUDIENCE: <strong style="color: #F5A623;">{mode.upper()}</strong></span>'
        f'<span>AUROC: <strong style="font-family: var(--font-mono); color: #E6EAF0;">{auroc:.1f}%</strong></span>'
        '</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

def render_kpi_card(title: str, value: str, subtext: str = "", border_color: str = None):
    """Renders a single KPI tile."""
    border_style = f"border-top: 3px solid {border_color};" if border_color else ""
    html = (
        f'<div class="scada-kpi" style="{border_style}">'
        f'<div class="scada-kpi-label">{title}</div>'
        f'<div class="scada-kpi-val">{value}</div>'
        f'<div class="scada-kpi-sub">{subtext}</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

def render_status_badge(status: str, label: str):
    """Renders a semantic status badge."""
    badge_cls = f"badge-{status}"
    return f'<span class="scada-badge {badge_cls}">{label}</span>'

def render_score_gauge(score: float, threshold: float, review_limit: float):
    """
    Renders a calibrated horizontal score gauge with three colored bands:
      - Pass zone (0 to review_limit)
      - Review band (review_limit to threshold)
      - Reject zone (threshold to max)
    Displays the score pin and verbal distance to limit.
    """
    max_scale = max(score * 1.25, threshold * 1.35, 12.0)
    score_pct = min(100.0, max(0.0, (score / max_scale) * 100.0))
    thresh_pct = min(100.0, max(0.0, (threshold / max_scale) * 100.0))
    review_pct = min(100.0, max(0.0, (review_limit / max_scale) * 100.0))

    # Calculate verbal distance note
    delta = score - threshold
    if delta >= 0:
        dist_note = f"<span style='color: #FF5C5C; font-weight: 600;'>+{abs(delta):.2f} above pass limit</span> (Action: Reject/Quarantine)"
    elif score >= review_limit:
        dist_note = f"<span style='color: #F5A623; font-weight: 600;'>{abs(delta):.2f} below pass limit</span> (Action: Inside review band, manual check required)"
    else:
        dist_note = f"<span style='color: #2BD4A0; font-weight: 600;'>{abs(delta):.2f} below pass limit</span> (Action: Pass nominal, within tolerance)"

    html = (
        '<div class="scada-gauge-container">'
        '<div style="display: flex; justify-content: space-between; font-size: 12px; color: var(--text-muted); font-family: var(--font-heading); text-transform: uppercase;">'
        f'<span>Nominal Pass Zone (0.00 – {review_limit:.2f})</span>'
        f'<span style="color: var(--accent-amber);">Review Band ({review_limit:.2f} – {threshold:.2f})</span>'
        f'<span style="color: var(--status-fail);">Reject Zone (≥ {threshold:.2f})</span>'
        '</div>'
        '<div style="position: relative; height: 16px; margin: 26px 0 16px 0; background-color: #1A212B; border-radius: 2px; border: 1px solid var(--border-subtle);">'
        f'<div style="position: absolute; left: 0; width: {review_pct}%; height: 100%; background-color: rgba(43, 212, 160, 0.45); border-right: 1px solid rgba(43, 212, 160, 0.7);"></div>'
        f'<div style="position: absolute; left: {review_pct}%; width: {thresh_pct - review_pct}%; height: 100%; background-color: rgba(245, 166, 35, 0.55); border-right: 1px solid rgba(245, 166, 35, 0.8);"></div>'
        f'<div style="position: absolute; left: {thresh_pct}%; width: {100.0 - thresh_pct}%; height: 100%; background-color: rgba(255, 92, 92, 0.45);"></div>'
        f'<div style="position: absolute; left: {score_pct}%; top: -8px; width: 4px; height: 32px; background-color: #FFFFFF; border: 1px solid #000; box-shadow: 0 0 6px rgba(255, 255, 255, 0.9); z-index: 10;">'
        f'<div style="position: absolute; top: -18px; left: 50%; transform: translateX(-50%); font-family: var(--font-mono); font-size: 11px; font-weight: 700; color: #FFFFFF; white-space: nowrap;">{score:.2f}</div>'
        '</div>'
        f'<div style="position: absolute; left: {thresh_pct}%; top: 16px; transform: translateX(-50%); font-family: var(--font-mono); font-size: 11px; color: #FF5C5C; white-space: nowrap;">Limit: {threshold:.2f}</div>'
        '</div>'
        '<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 14px; font-size: 13px;">'
        f'<div>Status: {dist_note}</div>'
        f'<div style="font-family: var(--font-mono); font-size: 12px; color: var(--text-muted);">Scale Range: [0.00 – {max_scale:.1f}]</div>'
        '</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)
    st.markdown(html, unsafe_allow_html=True)

def render_decision_card(decision: dict, score: float, threshold: float, true_label: str, mode: str):
    """Renders the decision box with honest false negative / false alarm feedback."""
    status = decision["status"]
    state = decision["state"]
    action_text = decision["action"]
    
    badge_html = render_status_badge(status, decision["label"])
    
    # Ground truth reconciliation logic
    mismatch_banner = ""
    if true_label == "Defective" and state == "PASS":
        mismatch_banner = (
            '<div style="background-color: rgba(245, 166, 35, 0.15); border: 1px solid #F5A623; padding: 10px 14px; border-radius: 2px; margin-top: 12px; color: #E6EAF0;">'
            '<strong style="color: #F5A623;">NOTICE: MODEL MISSED THIS DEFECT (FALSE NEGATIVE)</strong><br>'
            '<span style="font-size: 12px; color: #8B96A5;">The ground truth confirms this item is defective, but its subtle visual footprint scored below the pass limit. This is a known risk of global image-level pooling on small local flaws.</span>'
            '</div>'
        )
    elif true_label == "Normal" and state == "REJECT":
        mismatch_banner = (
            '<div style="background-color: rgba(76, 155, 255, 0.15); border: 1px solid #4C9BFF; padding: 10px 14px; border-radius: 2px; margin-top: 12px; color: #E6EAF0;">'
            '<strong style="color: #4C9BFF;">NOTICE: FALSE ALARM (FALSE POSITIVE)</strong><br>'
            '<span style="font-size: 12px; color: #8B96A5;">The ground truth confirms this item is normal, but natural surface variance exceeded the nominal limit. Manual inspection prevents unnecessary scrap.</span>'
            '</div>'
        )
    elif true_label == "Unknown":
        mismatch_banner = (
            '<div style="background-color: rgba(139, 150, 165, 0.15); border: 1px solid #8B96A5; padding: 8px 12px; border-radius: 2px; margin-top: 12px; color: #8B96A5; font-size: 12px;">'
            'Ground truth unverified (Custom Upload).'
            '</div>'
        )

    tech_details = ""
    if mode == "Engineer view":
        tech_details = (
            '<div style="margin-top: 12px; padding-top: 10px; border-top: 1px solid var(--border-subtle); font-family: var(--font-mono); font-size: 12px; color: var(--text-muted); display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">'
            f'<div>Raw Score: <strong style="color: #E6EAF0;">{score:.4f}</strong></div>'
            f'<div>Threshold (95th %ile): <strong style="color: #E6EAF0;">{threshold:.4f}</strong></div>'
            f'<div>Review Buffer Limit: <strong style="color: #E6EAF0;">{decision["review_limit"]:.4f}</strong></div>'
            f'<div>Buffer Margin: <strong style="color: #E6EAF0;">{REVIEW_BAND_MARGIN*100:.0f}%</strong></div>'
            '</div>'
        )

    html = (
        f'<div class="scada-card" style="border-left: 4px solid var(--status-{status});">'
        '<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">'
        '<span style="font-family: var(--font-heading); font-size: 14px; color: var(--text-muted); text-transform: uppercase;">AUTOMATED VERDICT</span>'
        f'{badge_html}'
        '</div>'
        f'<div style="font-size: 15px; font-weight: 600; color: #E6EAF0; margin-bottom: 4px;">{action_text}</div>'
        f'<div style="font-size: 12px; color: var(--text-muted);">Ground Truth Classification: <strong>{true_label}</strong></div>'
        f'{mismatch_banner}'
        f'{tech_details}'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

def render_glossary_expander():
    """Renders a collapsible glossary of terms."""
    with st.expander("REFERENCE GLOSSARY // TERMINOLOGY & ACRONYMS"):
        from ui.terms import TERMS
        glossary_rows = []
        for key, val in TERMS.items():
            glossary_rows.append({
                "Technical Term": val["engineer"],
                "Operator Translation": val["operator"],
                "Operational Definition": val["explanation"]
            })
        st.dataframe(pd.DataFrame(glossary_rows), use_container_width=True, hide_index=True)
