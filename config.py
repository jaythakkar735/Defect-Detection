"""
Industrial Anomaly Detection Configuration & Constants.
Defines UI design tokens, operational defaults, reliability grading thresholds,
and decision boundary margins.
"""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
ASSETS_DIR = PROJECT_ROOT / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"

# Category Definitions
CATEGORIES = ["toothbrush", "grid", "hazelnut", "leather", "wood", "zipper"]
DEFAULT_CATEGORY = "toothbrush"

# Factory Shift Simulation Defaults
DEFAULT_BATCH_SIZE = 1000
DEFAULT_REVIEW_CAPACITY_PCT = 10
DEFAULT_DEFECT_RATE_PCT = 5.0
DEFAULT_COST_REVIEW = 0.50
DEFAULT_COST_MISSED = 30.00
SIMULATION_N_TRIALS = 100
SIMULATION_SEED = 42

# Benchmark Standardized Parameters (Standardized fixed 10% budget across categories)
BENCHMARK_BATCH_SIZE = 1000
BENCHMARK_REVIEW_BUDGET_PCT = 10.0
BENCHMARK_DEFECT_RATE_PCT = 5.0
BENCHMARK_N_TRIALS = 50

# Decision Engine: 3-State Boundary
# Review band is relative: threshold * (1.0 - REVIEW_BAND_MARGIN) <= score < threshold
# REJECT is strictly score >= threshold.
# The review band is a display-only aid indicating borderline items.
REVIEW_BAND_MARGIN = 0.12

# Reliability Grading Constants (AUROC %)
# Strong: >= 90.0
# Moderate: 75.0 - 89.9
# Weak: 55.0 - 74.9
# Unreliable: < 55.0 or negative simulated financial savings
RELIABILITY_STRONG = 90.0
RELIABILITY_MODERATE = 75.0
RELIABILITY_WEAK = 55.0

def get_reliability_tier(auroc_pct: float, net_savings: float = 0.0) -> dict:
    """
    Grades category reliability based on AUROC and financial viability.
    Returns status label, semantic status key ('pass', 'review', 'fail'), and note.
    """
    if auroc_pct < RELIABILITY_WEAK or net_savings < 0:
        return {
            "tier": "Unreliable",
            "badge": "UNRELIABLE: WORSE THAN RANDOM",
            "status": "fail",
            "desc": "AUROC < 55% or negative cost savings. Global pooling fails on this structure; not recommended for deployment without localized patch-level feature banks."
        }
    elif auroc_pct < RELIABILITY_MODERATE:
        return {
            "tier": "Weak",
            "badge": "WEAK SEPARATION",
            "status": "review",
            "desc": f"AUROC {auroc_pct:.1f}% (55–74%). Subtle defects are partially diluted by normal pixels; requires high review budget to achieve high recall."
        }
    elif auroc_pct < RELIABILITY_STRONG:
        return {
            "tier": "Moderate",
            "badge": "MODERATE RELIABILITY",
            "status": "info",
            "desc": f"AUROC {auroc_pct:.1f}% (75–89%). Demonstrates consistent defect separation with minor overlap in fine edge cases."
        }
    else:
        return {
            "tier": "Strong",
            "badge": "STRONG DISCRIMINATION",
            "status": "pass",
            "desc": f"AUROC {auroc_pct:.1f}% (≥90%). Reliable separation between nominal production and defect population."
        }

def get_decision_state(score: float, threshold: float, margin_ratio: float = REVIEW_BAND_MARGIN) -> dict:
    """
    3-state decision engine:
      - PASS: score < threshold * (1 - margin_ratio)
      - SEND TO REVIEW: threshold * (1 - margin_ratio) <= score < threshold
      - REJECT: score >= threshold
    """
    review_limit = threshold * (1.0 - margin_ratio)
    
    if score >= threshold:
        return {
            "state": "REJECT",
            "label": "REJECT / QUARANTINE",
            "status": "fail",
            "action": "Quarantine product immediately. Critical anomaly detected above pass limit.",
            "review_limit": review_limit,
            "threshold": threshold
        }
    elif score >= review_limit:
        return {
            "state": "REVIEW",
            "label": "SEND TO HUMAN REVIEW",
            "status": "review",
            "action": "Route item to secondary visual review station. Anomaly score falls inside borderline review band.",
            "review_limit": review_limit,
            "threshold": threshold
        }
    else:
        return {
            "state": "PASS",
            "label": "AUTOMATIC PASS",
            "status": "pass",
            "action": "Release product to downstream packaging. Anomaly score is within nominal limits.",
            "review_limit": review_limit,
            "threshold": threshold
        }

# Design Tokens (Hex colors)
COLORS = {
    "bg": "#0E1116",
    "panel": "#151A21",
    "panel_raised": "#1B222B",
    "border": "#2A323D",
    "text": "#E6EAF0",
    "text_muted": "#8B96A5",
    "accent": "#F5A623",
    "pass": "#2BD4A0",
    "review": "#F5A623",
    "fail": "#FF5C5C",
    "info": "#4C9BFF"
}
