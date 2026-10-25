"""
Tab 2: Factory Shift Simulation (Human-in-the-Loop Review Budget).
Simulates real manufacturing shifts comparing Random Sampling vs. AI Prioritization,
computing defect capture rates, labor vs. escape costs, and financial sensitivity.
"""
import streamlit as st
import pandas as pd
from src.simulation import run_budget_simulation, compute_cost_model
from ui.charts import build_shift_comparison_chart, build_sensitivity_chart
from ui.terms import get_term_label, get_term_help
from config import SIMULATION_N_TRIALS, SIMULATION_SEED

def render_simulation_tab(
    normal_scores,
    defect_scores,
    batch_size: int,
    review_capacity_pct: int,
    defect_rate_pct: float,
    cost_review: float,
    cost_miss: float,
    threshold: float,
    selected_category: str,
    audience_mode: str
):
    st.markdown("### FACTORY SHIFT SIMULATION // FIXED REVIEW CAPACITY")

    review_budget = int(batch_size * (review_capacity_pct / 100.0))

    # How to read this page expander
    with st.expander("HOW TO EVALUATE SHIFT SIMULATION // ECONOMIC CONTEXT"):
        st.markdown(f"""
        - **Core Question:** In high-speed manufacturing, visual reviewers can only inspect **{review_budget:,} out of {batch_size:,} units** ({review_capacity_pct}% capacity). Does AI suspicion ranking catch more defects than taking {review_budget:,} random coupons?
        - **Budget Parity:** Both approaches conduct the exact same number of human reviews ({review_budget:,}). Reviewer labor cost is identical.
        - **Financial Value:** Savings stem entirely from preventing defective products from shipping to customers.
        """)

    # Run Monte Carlo simulation using empirical score distributions
    sim_results = run_budget_simulation(
        normal_scores=normal_scores,
        defective_scores=defect_scores,
        batch_size=batch_size,
        review_budget=review_budget,
        defect_prevalence=defect_rate_pct / 100.0,
        n_trials=SIMULATION_N_TRIALS,
        threshold=threshold,
        seed=SIMULATION_SEED
    )

    cost_results = compute_cost_model(
        sim_results,
        cost_per_review=cost_review,
        cost_per_missed_defect=cost_miss
    )

    n_defects = sim_results["n_defects"]
    found_a = sim_results["baseline_a"]["defects_caught_mean"]
    missed_a = sim_results["baseline_a"]["missed_defects"]
    recall_a = sim_results["baseline_a"]["recall_pct"]

    found_b = sim_results["approach_b"]["defects_caught_mean"]
    missed_b = sim_results["approach_b"]["missed_defects"]
    recall_b = sim_results["approach_b"]["recall_pct"]

    multiplier = sim_results["capture_multiplier"]
    extra_caught = max(0.0, found_b - found_a)

    # 1. Plain-English Scenario Sentence (Real Computed Numbers)
    st.markdown(f"""
    <div class="scada-scenario-box">
      <strong>Shift Scenario:</strong> In a production run of <strong>{batch_size:,} units</strong> with 
      <strong>{defect_rate_pct:.1f}% true defects ({n_defects:,} defective items)</strong>, factory inspectors 
      have time to manually review <strong>{review_budget:,} items</strong> ({review_capacity_pct}% review quota).<br>
      • <strong>Random Sampling (Status Quo)</strong> catches <strong>{found_a:.1f}</strong> of {n_defects} defects ({recall_a:.1f}% recall).<br>
      • <strong>AI-Assisted Prioritisation</strong> catches <strong>{found_b:.1f}</strong> of {n_defects} defects ({recall_b:.1f}% recall).
    </div>
    """, unsafe_allow_html=True)

    # 2. KPI Row
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        st.markdown(f"""
        <div class="scada-kpi">
          <div class="scada-kpi-label">TOTAL DEFECTS IN BATCH</div>
          <div class="scada-kpi-val">{n_defects:,} <span style="font-size: 14px; color: var(--text-muted);">units</span></div>
          <div class="scada-kpi-sub">Defect rate: {defect_rate_pct:.1f}% of {batch_size:,} total</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_col2:
        st.markdown(f"""
        <div class="scada-kpi">
          <div class="scada-kpi-label">RANDOM CHECKING RECALL</div>
          <div class="scada-kpi-val">{recall_a:.1f}%</div>
          <div class="scada-kpi-sub">{found_a:.1f} caught | {missed_a:.1f} escaped</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col3:
        status_b_border = "border-top: 3px solid #2BD4A0;" if recall_b >= recall_a else "border-top: 3px solid #FF5C5C;"
        st.markdown(f"""
        <div class="scada-kpi" style="{status_b_border}">
          <div class="scada-kpi-label">AI PRIORITISATION RECALL</div>
          <div class="scada-kpi-val">{recall_b:.1f}%</div>
          <div class="scada-kpi-sub">{found_b:.1f} caught | {missed_b:.1f} escaped</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col4:
        st.markdown(f"""
        <div class="scada-kpi" style="border-top: 3px solid var(--accent-amber);">
          <div class="scada-kpi-label">{get_term_label('capture_multiplier', audience_mode).upper()}</div>
          <div class="scada-kpi-val">{multiplier:.2f}x</div>
          <div class="scada-kpi-sub">+{extra_caught:.1f} extra defects caught</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 3. Horizontal Bar Chart (Direct value labels, distinct colors)
    chart_fig = build_shift_comparison_chart(found_a, missed_a, found_b, missed_b)
    st.plotly_chart(chart_fig, use_container_width=True, config={"displayModeBar": False})

    # 4. Financial Cost Model
    st.markdown("#### FINANCIAL QUALITY COST MODEL // LABOR VS. ESCAPED DEFECTS")
    
    fin_col1, fin_col2, fin_col3 = st.columns(3)
    with fin_col1:
        st.markdown(f"""
        <div class="scada-card">
          <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase;">Expected Cost: Random Checking</div>
          <div style="font-family: var(--font-mono); font-size: 22px; font-weight: 700; color: #E6EAF0;">${cost_results['baseline_a_total_cost']:.2f}</div>
          <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">Labor: ${cost_results['review_cost']:.2f} + Escapes: ${missed_a * cost_miss:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with fin_col2:
        st.markdown(f"""
        <div class="scada-card">
          <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase;">Expected Cost: AI Prioritisation</div>
          <div style="font-family: var(--font-mono); font-size: 22px; font-weight: 700; color: #E6EAF0;">${cost_results['approach_b_total_cost']:.2f}</div>
          <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">Labor: ${cost_results['review_cost']:.2f} + Escapes: ${missed_b * cost_miss:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with fin_col3:
        savings = cost_results["net_savings"]
        pct_savings = cost_results["pct_savings"]
        savings_color = "#2BD4A0" if savings >= 0 else "#FF5C5C"
        savings_prefix = "+$" if savings >= 0 else "-$"
        
        st.markdown(f"""
        <div class="scada-card" style="border-left: 3px solid {savings_color};">
          <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase;">Net Simulated Savings</div>
          <div style="font-family: var(--font-mono); font-size: 22px; font-weight: 700; color: {savings_color};">
            ${savings:.2f} ({pct_savings:.1f}%)
          </div>
          <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">Per shift of {batch_size:,} units</div>
        </div>
        """, unsafe_allow_html=True)

    # Plain statement of efficiency & when AI is worse than random
    if savings < 0 or multiplier < 1.0:
        st.warning(f"OPERATIONAL WARNING: On category '{selected_category.capitalize()}', AI prioritisation performs worse than random checking (Capture Multiplier {multiplier:.2f}x). Do not deploy without model retraining or localized patch feature banks.")
    else:
        savings_per_defect = (savings / extra_caught) if extra_caught > 0 else 0.0
        st.info(f"Efficiency metric: Each extra defect captured by AI triage delivers **${savings_per_defect:.2f} in net operational savings** under current cost assumptions.")

    # Mathematical Formula Expander for Engineer View
    if audience_mode == "Engineer view":
        with st.expander("TECHNICAL CALCULATION // COST FORMULA SPECIFICATION"):
            st.latex(r"\text{Total Expected Cost} = (B \times C_{\text{review}}) + (N_{\text{missed}} \times C_{\text{miss}})")
            st.markdown(f"""
            - **Review Budget ($B$):** `{review_budget:,}` inspections.
            - **Labor Cost ($C_{{\\text{{review}}}}$):** `${cost_review:.2f}` per inspected unit.
            - **Escaped Defect Penalty ($C_{{\\text{{miss}}}}$):** `${cost_miss:.2f}` per escaped flaw.
            - **Baseline A Escapes:** `{missed_a:.1f}` units $\\rightarrow$ `${missed_a * cost_miss:.2f}`
            - **Approach B Escapes:** `{missed_b:.1f}` units $\\rightarrow$ `${missed_b * cost_miss:.2f}`
            """)

    # 5. Sensitivity Analysis
    st.markdown("#### SENSITIVITY ANALYSIS // SAVINGS ACROSS DEFECT ESCAPE PENALTIES")
    
    penalties = [10.0, 25.0, 50.0, 75.0, 100.0, 150.0]
    sens_rows = []
    for p in penalties:
        cr = compute_cost_model(sim_results, cost_per_review=cost_review, cost_per_missed_defect=p)
        sens_rows.append({
            "penalty_num": p,
            "Escaped Defect Cost": f"${p:.2f}",
            "Random Checking Cost": f"${cr['baseline_a_total_cost']:.2f}",
            "AI Prioritisation Cost": f"${cr['approach_b_total_cost']:.2f}",
            "net_savings_num": cr["net_savings"],
            "Net Savings": f"${cr['net_savings']:.2f}",
            "Cost Cut": f"{cr['pct_savings']:.1f}%"
        })

    df_sens = pd.DataFrame(sens_rows)
    
    sens_col_left, sens_col_right = st.columns([1, 1], gap="medium")
    with sens_col_left:
        # Formatted table without horizontal scroll
        st.dataframe(
            df_sens[["Escaped Defect Cost", "Random Checking Cost", "AI Prioritisation Cost", "Net Savings", "Cost Cut"]],
            use_container_width=True,
            hide_index=True
        )
    with sens_col_right:
        # Sensitivity Chart
        sens_chart = build_sensitivity_chart(df_sens)
        st.plotly_chart(sens_chart, use_container_width=True, config={"displayModeBar": False})
