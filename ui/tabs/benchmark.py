"""
Tab 3: Multi-Category Benchmark & Cross-Domain Validation.
Standardizes on a fixed 10% review capacity across all 6 MVTec categories,
implements the 4-tier reliability grading system, and presents dual operator/engineer analyses.
"""
import streamlit as st
import pandas as pd
from config import (
    CATEGORIES,
    BENCHMARK_BATCH_SIZE,
    BENCHMARK_REVIEW_BUDGET_PCT,
    BENCHMARK_DEFECT_RATE_PCT,
    BENCHMARK_N_TRIALS,
    get_reliability_tier
)
from src.simulation import run_budget_simulation, compute_cost_model
from ui.charts import build_benchmark_auroc_chart

@st.cache_data
def compute_benchmark_table(_category_results: dict):
    """
    Computes cross-domain benchmark table using cached category evaluation outputs.
    Standardized across all categories at fixed 10% review capacity and $30.00 defect escape penalty.
    """
    rows = []
    for cat in CATEGORIES:
        res = _category_results[cat]
        ev = res["eval_res"]
        
        n_sc = ev["scores"][ev["labels"] == 0]
        d_sc = ev["scores"][ev["labels"] == 1]
        
        sim = run_budget_simulation(
            normal_scores=n_sc,
            defective_scores=d_sc,
            batch_size=BENCHMARK_BATCH_SIZE,
            review_budget=int(BENCHMARK_BATCH_SIZE * (BENCHMARK_REVIEW_BUDGET_PCT / 100.0)),
            defect_prevalence=BENCHMARK_DEFECT_RATE_PCT / 100.0,
            n_trials=BENCHMARK_N_TRIALS,
            seed=42
        )
        
        c_mod = compute_cost_model(sim, cost_per_review=0.50, cost_per_missed_defect=30.00)
        auroc_pct = ev["auroc"] * 100.0
        net_save = c_mod["net_savings"]
        tier_info = get_reliability_tier(auroc_pct, net_save)
        
        rows.append({
            "Category": cat.capitalize(),
            "Domain": "Texture" if cat in ["grid", "leather", "wood"] else "Object",
            "Test Set Size": len(ev["labels"]),
            "Image AUROC": auroc_pct,
            "Random Recall (10%)": sim["baseline_a"]["recall_pct"],
            "AI Recall (10%)": sim["approach_b"]["recall_pct"],
            "Capture Boost": sim["capture_multiplier"],
            "Net Savings ($)": net_save,
            "Reliability Grade": tier_info["badge"],
            "Tier": tier_info["tier"]
        })
        
    return pd.DataFrame(rows)

def render_benchmark_tab(category_cache: dict, audience_mode: str):
    st.markdown("### MULTI-CATEGORY BENCHMARK // CROSS-DOMAIN PORTFOLIO")

    # Policy note on 10% budget vs dynamic slider
    st.markdown("""
    <div class="scada-scenario-box" style="border-left-color: #4C9BFF;">
      <strong>Standardized Methodology Note:</strong> While Tab 2 allows interactive review budget experimentation, 
      this benchmark table intentionally fixes the human inspection budget at <strong>10% (100 reviews per 1,000 units)</strong> 
      across all six categories. This establishes an objective, apples-to-apples basis for cross-domain comparison.
    </div>
    """, unsafe_allow_html=True)

    df_bench = compute_benchmark_table(category_cache)

    # 1. Ranked AUROC Chart
    df_sorted = df_bench.sort_values(by="Image AUROC", ascending=False)
    chart = build_benchmark_auroc_chart(
        categories=df_sorted["Category"].tolist(),
        aurocs=df_sorted["Image AUROC"].tolist(),
        tiers=df_sorted["Tier"].tolist()
    )
    st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})

    # 2. Benchmark Table with column_config styling
    st.dataframe(
        df_bench[[
            "Category", "Domain", "Test Set Size", "Image AUROC", 
            "Random Recall (10%)", "AI Recall (10%)", "Capture Boost", 
            "Net Savings ($)", "Reliability Grade"
        ]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Image AUROC": st.column_config.ProgressColumn(
                "Image AUROC",
                format="%.1f%%",
                min_value=0,
                max_value=100
            ),
            "Random Recall (10%)": st.column_config.NumberColumn(format="%.1f%%"),
            "AI Recall (10%)": st.column_config.NumberColumn(format="%.1f%%"),
            "Capture Boost": st.column_config.NumberColumn(format="%.2fx"),
            "Net Savings ($)": st.column_config.NumberColumn(format="$%.2f")
        }
    )

    # Note on sample size confidence intervals
    st.caption("Note on Dataset Scale: Several categories (e.g., Toothbrush with 42 test images) have small sample counts, leading to wide confidence intervals. Reported metrics represent empirical performance on the benchmark split.")

    st.markdown("---")

    # 3. Dual Takeaway Panels: Operator View vs. Engineer View
    col_left, col_right = st.columns(2, gap="medium")

    with col_left:
        st.markdown("#### WHAT THIS MEANS FOR THE FACTORY")
        st.markdown("""
        * **Surface Textures (Leather, Wood):** Highly reliable. The model catches over 70% of defects within the top 10% of items checked, delivering predictable quality savings.
        * **Molded Objects (Zipper, Hazelnut):** Strong defect capture boost (5x to 7x over random checking). Suitable for live production triage.
        * **Fine Structured Parts (Toothbrush):** Moderate separation (73.9% AUROC). Many subtle bristle flaws are detected, but a 10% budget is insufficient to catch all defects.
        * **Repetitive Meshes (Grid):** **Unreliable.** The model cannot reliably distinguish bent wires from normal mesh using global image averaging. Manual inspection or specialized sensors remain required.
        """)

    with col_right:
        st.markdown("#### TECHNICAL NOTES // ARCHITECTURAL INSIGHTS")
        st.markdown("""
        * **Global Feature Dilution:** The baseline extracts 512-D vectors via `AdaptiveAvgPool2d` on ResNet-18. When a defect spans >1% of pixels (Leather cuts, Hazelnut cracks), it strongly displaces the global centroid.
        * **The Grid Failure Mode (Likely Cause):** Grid consists of a repetitive, high-frequency wire mesh. A single warped wire occupies <0.1% of the spatial area. Global spatial pooling averages the flaw into the background, yielding an AUROC of 46.0% (sub-random).
        * **Upgrade Path:** Overcoming this limitation requires moving from global image embeddings to patch-level memory banks (e.g., PatchCore) or self-attention architectures that preserve spatial coordinates.
        """)

    # 4. Known Limitations Section
    with st.expander("DOCUMENTED SYSTEM LIMITATIONS & FACTORY DEPLOYMENT PREREQUISITES"):
        st.markdown("""
        1. **Controlled Benchmark vs. Live Factory:** MVTec AD features fixed camera positions, uniform black backgrounds, and consistent lighting. Real production lines experience camera vibration, dust, and lighting drift that require continuous threshold recalibration.
        2. **Sample Size Constraints:** Toothbrush has only 12 normal and 30 defective test samples. While bootstrapping simulates 1,000-unit shifts statistically, physical validation on thousands of parts is necessary before deploying as an automated gate.
        3. **Display-Only Review Band:** The 12% review buffer is designed to assist human operator attention; the underlying algorithmic score remains continuous.
        """)
