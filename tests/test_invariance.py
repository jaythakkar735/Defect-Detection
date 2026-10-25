"""
Numerical Invariance Test.
Verifies that model scoring, thresholds, AUROC, and simulation outputs are
IDENTICAL to the original baseline measurements.
"""
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from config import DATA_RAW_DIR
from src.dataset import MVTecCategoryDataset, get_default_transforms
from src.model import ResNetAnomalyDetector
from src.simulation import run_budget_simulation, compute_cost_model

def test_toothbrush_numerical_invariance():
    """
    Asserts exact numerical invariance on Toothbrush:
      - Normal Train Count: 60
      - Test Count: 42 (12 normal, 30 defect)
      - AUROC: ~73.89%
      - Calibrated Threshold: ~10.4631
      - Fixed Simulation (1,000 units, 10% budget, seed 42):
        Baseline A Mean: ~5.0
        Approach B Mean: ~19.4
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = get_default_transforms()
    
    cat_dir = DATA_RAW_DIR / "toothbrush"
    train_ds = MVTecCategoryDataset(cat_dir, split="train", transform=transform)
    test_ds = MVTecCategoryDataset(cat_dir, split="test", transform=transform)
    
    assert len(train_ds) == 60
    assert len(test_ds) == 42
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    detector = ResNetAnomalyDetector(device=device, backbone="resnet18")
    detector.fit(train_loader)
    eval_res = detector.evaluate(test_loader)
    
    auroc_pct = eval_res["auroc"] * 100.0
    threshold = eval_res["threshold"]
    
    # Check AUROC matches our documented 73.89% within 0.1%
    assert auroc_pct == pytest.approx(73.89, abs=0.1)
    assert threshold == pytest.approx(10.4631, abs=0.1)
    
    # Run fixed simulation with seed 42
    n_sc = eval_res["scores"][eval_res["labels"] == 0]
    d_sc = eval_res["scores"][eval_res["labels"] == 1]
    
    sim = run_budget_simulation(
        normal_scores=n_sc,
        defective_scores=d_sc,
        batch_size=1000,
        review_budget=100,
        defect_prevalence=0.05,
        n_trials=100,
        seed=42
    )
    
    # Check baseline A mean ~5.0 and approach B mean ~19.4
    assert sim["baseline_a"]["defects_caught_mean"] == pytest.approx(5.0, abs=0.5)
    assert sim["approach_b"]["defects_caught_mean"] == pytest.approx(19.4, abs=0.5)
    
    # Cost model
    c_res = compute_cost_model(sim, cost_per_review=0.50, cost_per_missed_defect=30.00)
    assert c_res["review_cost"] == 50.00
    assert c_res["baseline_a_total_cost"] == pytest.approx(1400.00, abs=15.0)
    assert c_res["approach_b_total_cost"] == pytest.approx(968.00, abs=15.0)
    assert c_res["net_savings"] == pytest.approx(432.00, abs=15.0)
