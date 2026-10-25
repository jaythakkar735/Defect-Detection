"""
Unit tests for factory shift simulation and financial cost models.
"""
import numpy as np
import pytest
from src.simulation import run_budget_simulation, compute_cost_model

def test_simulation_random_sampling_expected_value():
    """
    Random sampling on a batch of 1,000 items with 5% defects (50 defects)
    and a 10% review capacity (100 reviews) has a theoretical expected value of
    E[X] = 50 * 0.10 = 5.0 defects.
    """
    normal_scores = np.array([7.0, 7.5, 8.0, 8.5, 9.0])
    defect_scores = np.array([12.0, 13.0, 14.0, 15.0])
    
    sim = run_budget_simulation(
        normal_scores=normal_scores,
        defective_scores=defect_scores,
        batch_size=1000,
        review_budget=100,
        defect_prevalence=0.05,
        n_trials=100,
        seed=42
    )
    
    mean_a = sim["baseline_a"]["defects_caught_mean"]
    assert sim["n_defects"] == 50
    # Over 100 trials, mean random defects caught should be ~5.0 within statistical tolerance
    assert 4.0 <= mean_a <= 6.0

def test_simulation_perfect_ai_separation():
    """
    When defect scores are completely separable from normal scores,
    reviewing the top 100 highest scores out of 1,000 should capture ALL 50 defects (100% recall).
    """
    normal_scores = np.array([1.0, 2.0, 3.0])
    defect_scores = np.array([10.0, 11.0, 12.0])
    
    sim = run_budget_simulation(
        normal_scores=normal_scores,
        defective_scores=defect_scores,
        batch_size=1000,
        review_budget=100,  # 100 reviews can easily capture all 50 defects
        defect_prevalence=0.05,
        n_trials=10,
        seed=42
    )
    
    mean_b = sim["approach_b"]["defects_caught_mean"]
    recall_b = sim["approach_b"]["recall_pct"]
    assert mean_b == 50.0
    assert recall_b == 100.0

def test_cost_model_arithmetic():
    """Verify exact financial calculations for labor, escapes, and net savings."""
    sim_dummy = {
        "review_budget": 100,
        "baseline_a": {"missed_defects": 45.0},
        "approach_b": {"missed_defects": 30.6}
    }
    
    cost_res = compute_cost_model(
        sim_dummy,
        cost_per_review=0.50,
        cost_per_missed_defect=30.00
    )
    
    # Review labor cost = 100 * 0.50 = $50.00
    assert cost_res["review_cost"] == 50.00
    
    # Baseline A: 50.00 + (45.0 * 30.00) = 50.00 + 1350.00 = 1400.00
    assert cost_res["baseline_a_total_cost"] == 1400.00
    
    # Approach B: 50.00 + (30.6 * 30.00) = 50.00 + 918.00 = 968.00
    assert cost_res["approach_b_total_cost"] == pytest.approx(968.00, abs=1e-2)
    
    # Net savings = 1400.00 - 968.00 = 432.00
    assert cost_res["net_savings"] == pytest.approx(432.00, abs=1e-2)
    
    # Pct savings = (432.00 / 1400.00) * 100 = 30.857%
    assert cost_res["pct_savings"] == pytest.approx(30.857, abs=1e-2)
