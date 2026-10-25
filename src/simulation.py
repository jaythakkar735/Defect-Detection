import numpy as np
import pandas as pd

def run_budget_simulation(
    normal_scores: np.ndarray,
    defective_scores: np.ndarray,
    batch_size: int = 1000,
    review_budget: int = 100,
    defect_prevalence: float = 0.05,
    n_trials: int = 100,
    threshold: float = None,
    seed: int = 42
) -> dict:
    """
    Simulates factory shifts comparing:
      - Baseline A: Random Inspection of `review_budget` units.
      - Approach B: AI Prioritization (Top `review_budget` highest scores).
      - Approach C: Threshold-based Alerting (All units >= threshold).
    """
    np.random.seed(seed)
    n_defects = int(batch_size * defect_prevalence)
    n_normals = batch_size - n_defects

    caught_a_list = []
    caught_b_list = []
    caught_c_list = []
    alerts_c_list = []

    for _ in range(n_trials):
        # Bootstrap sample from empirical test distributions
        batch_norm = np.random.choice(normal_scores, size=n_normals, replace=True)
        batch_defect = np.random.choice(defective_scores, size=n_defects, replace=True)

        batch_scores = np.concatenate([batch_norm, batch_defect])
        batch_labels = np.concatenate([np.zeros(n_normals), np.ones(n_defects)])

        # 1. Baseline A: Random Sampling
        rand_idx = np.random.choice(batch_size, size=review_budget, replace=False)
        caught_a = int(np.sum(batch_labels[rand_idx] == 1))
        caught_a_list.append(caught_a)

        # 2. Approach B: AI Prioritization (Fixed review budget)
        top_idx = np.argsort(batch_scores)[-review_budget:]
        caught_b = int(np.sum(batch_labels[top_idx] == 1))
        caught_b_list.append(caught_b)

        # 3. Approach C: Threshold screening (Variable review workload)
        if threshold is not None:
            flagged = batch_scores >= threshold
            alerts_c = int(np.sum(flagged))
            caught_c = int(np.sum(batch_labels[flagged] == 1))
            caught_c_list.append(caught_c)
            alerts_c_list.append(alerts_c)

    mean_a = float(np.mean(caught_a_list))
    std_a = float(np.std(caught_a_list))
    recall_a = (mean_a / n_defects) * 100.0

    mean_b = float(np.mean(caught_b_list))
    std_b = float(np.std(caught_b_list))
    recall_b = (mean_b / n_defects) * 100.0

    multiplier = (mean_b / mean_a) if mean_a > 0 else 1.0

    results = {
        "batch_size": batch_size,
        "n_defects": n_defects,
        "review_budget": review_budget,
        "defect_prevalence": defect_prevalence,
        "baseline_a": {"defects_caught_mean": mean_a, "defects_caught_std": std_a, "recall_pct": recall_a, "missed_defects": n_defects - mean_a},
        "approach_b": {"defects_caught_mean": mean_b, "defects_caught_std": std_b, "recall_pct": recall_b, "missed_defects": n_defects - mean_b},
        "capture_multiplier": multiplier
    }

    if threshold is not None:
        mean_c = float(np.mean(caught_c_list))
        alerts_c = float(np.mean(alerts_c_list))
        recall_c = (mean_c / n_defects) * 100.0
        results["approach_c"] = {
            "threshold": threshold,
            "defects_caught_mean": mean_c,
            "avg_alerts_triggered": alerts_c,
            "recall_pct": recall_c,
            "missed_defects": n_defects - mean_c
        }

    return results

def compute_cost_model(
    simulation_results: dict,
    cost_per_review: float = 0.50,
    cost_per_missed_defect: float = 30.00
) -> dict:
    """Computes total expected operational cost under explicit financial assumptions."""
    b_budget = simulation_results["review_budget"]
    missed_a = simulation_results["baseline_a"]["missed_defects"]
    missed_b = simulation_results["approach_b"]["missed_defects"]

    cost_review = b_budget * cost_per_review
    cost_missed_a = missed_a * cost_per_missed_defect
    cost_missed_b = missed_b * cost_per_missed_defect

    total_a = cost_review + cost_missed_a
    total_b = cost_review + cost_missed_b
    net_savings = total_a - total_b
    pct_savings = (net_savings / total_a) * 100.0 if total_a > 0 else 0.0

    return {
        "cost_per_review": cost_per_review,
        "cost_per_missed_defect": cost_per_missed_defect,
        "review_cost": cost_review,
        "baseline_a_total_cost": total_a,
        "approach_b_total_cost": total_b,
        "net_savings": net_savings,
        "pct_savings": pct_savings
    }
