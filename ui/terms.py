"""
Single translation and glossary dictionary for Audience Mode:
Operator View (Plain language, decision-oriented) vs. Engineer View (Technical ML metrics).
"""

TERMS = {
    "anomaly_score": {
        "operator": "Suspicion Score",
        "engineer": "Euclidean Distance Score (L2 to Centroid)",
        "explanation": "Numerical measurement of how much an item deviates from normal products (higher = more unusual)."
    },
    "threshold": {
        "operator": "Pass / Fail Limit",
        "engineer": "Nominal Calibration Threshold (95th %ile)",
        "explanation": "The score limit above which an item is flagged for inspection."
    },
    "review_band": {
        "operator": "Review Zone Margin",
        "engineer": "Borderline Review Band (12% buffer below threshold)",
        "explanation": "Items falling just below the pass/fail limit where human verification is recommended."
    },
    "recall_at_budget": {
        "operator": "Defects Found @ Inspection Capacity",
        "engineer": "Recall @ Fixed Inspection Budget (B)",
        "explanation": "The percentage of true factory defects captured within the affordable inspection quota."
    },
    "auroc": {
        "operator": "Overall Ranking Quality",
        "engineer": "Area Under ROC Curve (Image-Level AUROC)",
        "explanation": "Overall discrimination accuracy across all thresholds (50% = guessing, 100% = flawless separation)."
    },
    "capture_multiplier": {
        "operator": "Defect Detection Boost",
        "engineer": "Relative Defect Capture Multiplier (AI vs Random)",
        "explanation": "How many times more defects the AI triage finds compared to status-quo random checking."
    },
    "escaped_defect": {
        "operator": "Missed Customer Defect",
        "engineer": "False Negative Escapee",
        "explanation": "A defective item that bypassed inspection and reached downstream packaging or customers."
    },
    "false_alarm": {
        "operator": "False Alarm",
        "engineer": "False Positive (Type I Error)",
        "explanation": "A normal part that was flagged unnecessarily for human inspection."
    },
    "missed_defect": {
        "operator": "Missed Defect",
        "engineer": "False Negative (Type II Error)",
        "explanation": "A defective part that was mistakenly allowed to pass."
    },
    "one_class_training": {
        "operator": "Learning from Perfect Parts Only",
        "engineer": "One-Class Unsupervised Feature Centroid",
        "explanation": "Model learns what normal looks like because factory defects are too rare to train standard classifiers."
    }
}

def get_term_label(key: str, mode: str = "Operator view") -> str:
    """Returns the plain or technical label for a given term."""
    item = TERMS.get(key, {})
    if mode == "Engineer view":
        return item.get("engineer", key.replace("_", " ").title())
    return item.get("operator", key.replace("_", " ").title())

def get_term_help(key: str) -> str:
    """Returns the one-sentence tooltip explanation."""
    return TERMS.get(key, {}).get("explanation", "")
