"""
Plotly Chart Builders for the SCADA Design System.
Enforces dark graphite palette, direct value labeling, tabular numerals, and zero truncated axes.
"""
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from config import COLORS

LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=COLORS["panel_raised"],
    font=dict(family="IBM Plex Sans, sans-serif", color=COLORS["text"], size=12),
    margin=dict(l=10, r=20, t=35, b=25),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=11, color=COLORS["text_muted"])
    )
)

def build_shift_comparison_chart(found_a: float, missed_a: float, found_b: float, missed_b: float) -> go.Figure:
    """
    Builds a clean horizontal stacked bar chart:
    Comparing Random Checking vs. AI-Prioritized Checking.
    Displays direct integer/float annotations on segments without truncation.
    """
    categories = ["Random Checking", "AI-Prioritised Checking"]
    found_vals = [round(found_a, 1), round(found_b, 1)]
    missed_vals = [round(missed_a, 1), round(missed_b, 1)]

    fig = go.Figure()

    # Found defects (Status Pass green)
    fig.add_trace(go.Bar(
        y=categories,
        x=found_vals,
        name="Defects Found (Inspected)",
        orientation="h",
        marker=dict(color=COLORS["pass"], line=dict(color=COLORS["border"], width=1)),
        text=[f"{v:.1f} caught" for v in found_vals],
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(family="JetBrains Mono, monospace", color="#0E1116", size=12, weight="bold")
    ))

    # Missed defects (Status Fail muted red)
    fig.add_trace(go.Bar(
        y=categories,
        x=missed_vals,
        name="Missed / Escaped Defects",
        orientation="h",
        marker=dict(color="rgba(255, 92, 92, 0.65)", line=dict(color=COLORS["border"], width=1)),
        text=[f"{v:.1f} escaped" for v in missed_vals],
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(family="JetBrains Mono, monospace", color="#FFFFFF", size=12, weight="bold")
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        barmode="stack",
        height=210,
        title=dict(
            text="DEFECT CAPTURE // FOUND VS. MISSED (1,000 UNITS)",
            font=dict(family="Barlow Condensed, sans-serif", size=14, color=COLORS["text_muted"])
        ),
        xaxis=dict(
            title=dict(text="Total True Defects in Batch", font=dict(size=11, color=COLORS["text_muted"])),
            gridcolor=COLORS["border"],
            zerolinecolor=COLORS["border"]
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(family="Barlow Condensed, sans-serif", size=13, color=COLORS["text"])
        )
    )
    return fig

def build_sensitivity_chart(df_sensitivity: pd.DataFrame) -> go.Figure:
    """
    Renders savings vs. escaped defect cost as a clean line + scatter graph.
    Eliminates horizontal scrolling issues while highlighting positive vs negative ROI.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_sensitivity["penalty_num"],
        y=df_sensitivity["net_savings_num"],
        mode="lines+markers",
        name="Net Simulated Savings",
        line=dict(color=COLORS["accent"], width=2.5),
        marker=dict(size=8, color=COLORS["accent"], line=dict(color="#FFFFFF", width=1.5)),
        hovertemplate="Penalty: $%{x:.0f}<br>Net Savings: $%{y:.2f}<extra></extra>"
    ))

    # Zero savings reference line
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["border"], annotation_text="Break-even", annotation_position="bottom right")

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        height=240,
        title=dict(
            text="ECONOMIC SENSITIVITY // NET SAVINGS VS. ESCAPED DEFECT COST",
            font=dict(family="Barlow Condensed, sans-serif", size=14, color=COLORS["text_muted"])
        ),
        xaxis=dict(
            title=dict(text="Cost of One Escaped Defect ($)", font=dict(size=11, color=COLORS["text_muted"])),
            gridcolor=COLORS["border"],
            tickprefix="$",
            zerolinecolor=COLORS["border"]
        ),
        yaxis=dict(
            title=dict(text="Net Savings per Shift ($)", font=dict(size=11, color=COLORS["text_muted"])),
            gridcolor=COLORS["border"],
            tickprefix="$",
            zerolinecolor=COLORS["border"]
        )
    )
    return fig

def build_benchmark_auroc_chart(categories: list, aurocs: list, tiers: list) -> go.Figure:
    """Renders ranked AUROC bar chart with 50% random-guess reference line."""
    # Color bars by reliability tier
    colors = []
    for t in tiers:
        if t == "Strong":
            colors.append(COLORS["pass"])
        elif t == "Moderate":
            colors.append(COLORS["info"])
        elif t == "Weak":
            colors.append(COLORS["review"])
        else:
            colors.append(COLORS["fail"])

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=categories,
        y=aurocs,
        marker=dict(color=colors, line=dict(color=COLORS["border"], width=1)),
        text=[f"{v:.1f}%" for v in aurocs],
        textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", size=11, color=COLORS["text"]),
        hovertemplate="Category: %{x}<br>AUROC: %{y:.2f}%<extra></extra>"
    ))

    # Random guess benchmark line (50%)
    fig.add_hline(
        y=50.0,
        line_dash="dot",
        line_color=COLORS["fail"],
        line_width=1.5,
        annotation_text="Random Guess (50.0%)",
        annotation_font=dict(color=COLORS["fail"], size=10),
        annotation_position="bottom right"
    )

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        height=280,
        title=dict(
            text="CROSS-DOMAIN DISCRIMINATION ACCURACY (IMAGE-LEVEL AUROC)",
            font=dict(family="Barlow Condensed, sans-serif", size=14, color=COLORS["text_muted"])
        ),
        xaxis=dict(
            gridcolor=COLORS["border"],
            tickfont=dict(family="Barlow Condensed, sans-serif", size=13)
        ),
        yaxis=dict(
            range=[0, 108],
            title=dict(text="AUROC (%)", font=dict(size=11, color=COLORS["text_muted"])),
            gridcolor=COLORS["border"]
        )
    )
    return fig

def build_roc_curve_chart(fpr: np.ndarray, tpr: np.ndarray, auroc: float) -> go.Figure:
    """Engineer View: Plots the Receiver Operating Characteristic curve."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode="lines",
        name=f"ResNet-18 (AUC = {auroc:.3f})",
        line=dict(color=COLORS["accent"], width=2.5)
    ))

    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random Guess (AUC = 0.500)",
        line=dict(color=COLORS["text_muted"], width=1.5, dash="dash")
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        height=260,
        title=dict(
            text="ROC CURVE // DISCRIMINATION TRADE-OFF",
            font=dict(family="Barlow Condensed, sans-serif", size=13, color=COLORS["text_muted"])
        ),
        xaxis=dict(
            title=dict(text="False Positive Rate (FPR)", font=dict(size=11, color=COLORS["text_muted"])),
            range=[0, 1.02],
            gridcolor=COLORS["border"]
        ),
        yaxis=dict(
            title=dict(text="True Positive Rate / Recall", font=dict(size=11, color=COLORS["text_muted"])),
            range=[0, 1.02],
            gridcolor=COLORS["border"]
        )
    )
    return fig
