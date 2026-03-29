# server/tasks/task_hard.py
HARD_TASK = {
    "task_id": "task_hard",
    "difficulty": "hard",
    "description": (
        "An e-commerce platform's Trust & Safety policy framework is underperforming. "
        "Historical simulation data shows high false positive rates (legitimate sellers "
        "being flagged) and missed fraud cases. Evolve the policy framework to improve "
        "both precision and recall while maintaining seller trust. You must propose "
        "modifications to at least 2 existing policies and justify trade-offs."
    ),
    "data_corpus": [
        {"id": "seller_001", "type": "legitimate", "flags": ["new_account", "high_volume"], "outcome": "wrongly_suspended"},
        {"id": "seller_002", "type": "fraudulent", "flags": ["price_manipulation"], "outcome": "missed"},
        {"id": "seller_003", "type": "legitimate", "flags": ["foreign_bank"], "outcome": "wrongly_suspended"},
        {"id": "seller_004", "type": "fraudulent", "flags": ["fake_reviews", "new_account"], "outcome": "correctly_caught"},
        {"id": "seller_005", "type": "legitimate", "flags": ["high_returns"], "outcome": "wrongly_suspended"},
        {"id": "seller_006", "type": "fraudulent", "flags": ["stolen_card_payments"], "outcome": "missed"},
        {"id": "seller_007", "type": "fraudulent", "flags": ["counterfeit_goods"], "outcome": "missed"},
        {"id": "seller_008", "type": "legitimate", "flags": ["seasonal_spike"], "outcome": "wrongly_suspended"},
    ],
    "current_policies": [
        {"id": "ts_pol_001", "text": "Any new seller account with more than 50 transactions in the first week will be suspended for review."},
        {"id": "ts_pol_002", "text": "Sellers with a return rate above 15% will be flagged for investigation."},
        {"id": "ts_pol_003", "text": "Sellers using non-domestic bank accounts will require manual approval."},
        {"id": "ts_pol_004", "text": "Any account with 3 or more fraud reports in 30 days will be permanently banned."},
        {"id": "ts_pol_005", "text": "Price changes of more than 20% within 24 hours will trigger an automatic hold."},
        {"id": "ts_pol_006", "text": "Sellers receiving 5+ negative reviews in 7 days will be suspended pending review."},
    ],
    "policy_outcomes": [
        {"metric": "false_positive_rate", "value": 0.42, "target": 0.10},
        {"metric": "fraud_detection_rate", "value": 0.31, "target": 0.85},
        {"metric": "seller_trust_score", "value": 0.54, "target": 0.80},
        {"metric": "review_queue_overload", "value": 0.89, "target": 0.30},
        {"metric": "legitimate_revenue_lost", "value": 0.28, "target": 0.05},
    ],
    "system_metrics": {
        "false_positive_rate": 0.42,
        "fraud_detection_rate": 0.31,
        "seller_trust_score": 0.54,
        "review_queue_overload": 0.89,
    },
    "identified_issues": [
        {"issue": "Blanket new-account volume rule catches legitimate seasonal sellers"},
        {"issue": "Return rate threshold doesn't distinguish category (electronics vs. fashion)"},
        {"issue": "Manual approval bottleneck creates 14-day delays for legitimate foreign sellers"},
    ],
    "num_policies": 6,
    "num_data_points": 8,
}
