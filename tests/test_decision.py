"""
Unit tests for 3-state decision engine and reliability grading.
"""
import pytest
from config import (
    get_decision_state,
    get_reliability_tier,
    REVIEW_BAND_MARGIN,
    RELIABILITY_STRONG,
    RELIABILITY_MODERATE,
    RELIABILITY_WEAK
)

def test_decision_state_toothbrush_sample():
    """
    Test requirement 2:
    Using Toothbrush sample (score ~8.98, threshold ~10.46).
    With relative margin of 12%:
      review_limit = 10.46 * (1 - 0.12) = 9.2048
      Since 8.98 < 9.2048, item lands in 'PASS'.
    """
    score = 8.98
    threshold = 10.46
    margin = 0.12
    
    decision = get_decision_state(score, threshold, margin_ratio=margin)
    
    assert decision["state"] == "PASS"
    assert decision["review_limit"] == pytest.approx(9.2048, abs=1e-4)
    assert decision["status"] == "pass"

def test_decision_state_review_band():
    """Test sample falling inside the 12% review band."""
    score = 9.50
    threshold = 10.46
    margin = 0.12  # review_limit is 9.2048
    
    decision = get_decision_state(score, threshold, margin_ratio=margin)
    assert decision["state"] == "REVIEW"
    assert decision["status"] == "review"

def test_decision_state_reject():
    """Test sample exceeding threshold."""
    score = 11.20
    threshold = 10.46
    
    decision = get_decision_state(score, threshold)
    assert decision["state"] == "REJECT"
    assert decision["status"] == "fail"

def test_reliability_tiers():
    """
    Test requirement 1:
    - Strong: >= 90
    - Moderate: 75 to 89.9 (Hazelnut at 84.71% must be Moderate!)
    - Weak: 55 to 74.9 (Toothbrush at 73.89% is Weak)
    - Unreliable: < 55 or negative savings (Grid at 46.03% is Unreliable)
    """
    # Strong
    strong_tier = get_reliability_tier(92.5)
    assert strong_tier["tier"] == "Strong"
    assert strong_tier["status"] == "pass"

    # Moderate (Hazelnut)
    hazelnut_tier = get_reliability_tier(84.71)
    assert hazelnut_tier["tier"] == "Moderate"
    assert hazelnut_tier["status"] == "info"

    # Weak (Toothbrush)
    toothbrush_tier = get_reliability_tier(73.89)
    assert toothbrush_tier["tier"] == "Weak"
    assert toothbrush_tier["status"] == "review"

    # Unreliable (Grid)
    grid_tier = get_reliability_tier(46.03)
    assert grid_tier["tier"] == "Unreliable"
    assert grid_tier["status"] == "fail"
    assert "UNRELIABLE" in grid_tier["badge"]
