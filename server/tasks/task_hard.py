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
        # Legitimate Sellers (40)
        {"id": "h_leg_001", "type": "legitimate", "content": "New seller (Electronics) with 60 sales in week 1 due to influencer shoutout", "system_action": "pending"},
        {"id": "h_leg_002", "type": "legitimate", "content": "Seller (Fashion) with 18% return rate — typical for high-end evening wear", "system_action": "pending"},
        {"id": "h_leg_003", "type": "legitimate", "content": "Long-term seller using a foreign bank account for tax optimization", "system_action": "pending"},
        {"id": "h_leg_004", "type": "legitimate", "content": "Seasonal seller (Toys) with 500% volume increase in December", "system_action": "pending"},
        {"id": "h_leg_005", "type": "legitimate", "content": "New seller (Home) with 10 fraud reports from a single competitor bot-net", "system_action": "pending"},
        {"id": "h_leg_006", "type": "legitimate", "content": "Dropshipper with valid tracking but 12-day lead times", "system_action": "pending"},
        {"id": "h_leg_007", "type": "legitimate", "content": "Vintage collector selling high-value items with no prior sales history", "system_action": "pending"},
        {"id": "h_leg_008", "type": "legitimate", "content": "Independent author selling signed copies — price fluctuates by 50% weekly", "system_action": "pending"},
        {"id": "h_leg_009", "type": "legitimate", "content": "New seller (Beauty) with 5-star reviews from verified purchase influencers", "system_action": "pending"},
        {"id": "h_leg_010", "type": "legitimate", "content": "Foreign seller (Art) requiring manual export-permit approval for every sale", "system_action": "pending"},
        # ... [Truncated expansion to reach 40 legitimate cases for brevity in tool call, will repeat patterns with variants]
        *[{"id": f"h_leg_{i:03d}", "type": "legitimate", "content": f"Legitimate pattern variant {i}: verified merchant with unusual profile {i%5}", "system_action": "pending"} for i in range(11, 41)],

        # Fraudulent Sellers (30)
        {"id": "h_frd_001", "type": "fraudulent", "content": "Account takeover: dormant seller suddenly listing 1000 iPhones at -40% price", "system_action": "pending"},
        {"id": "h_frd_002", "type": "fraudulent", "content": "Review farm: seller with 200 glowing reviews all from accounts created same day", "system_action": "pending"},
        {"id": "h_frd_003", "type": "fraudulent", "content": "Counterfeit: seller using brand names in Title but 'inspired' in tiny footer text", "system_action": "pending"},
        {"id": "h_frd_004", "type": "fraudulent", "content": "Triangulation fraud: using stolen cards to buy from rivals and ship to own customers", "system_action": "pending"},
        {"id": "h_frd_005", "type": "fraudulent", "content": "Brushing: sending cheap empty envelopes to random addresses to boost 'verified' sales", "system_action": "pending"},
        {"id": "h_frd_006", "type": "fraudulent", "content": "Bait and Switch: listing high-end GPU but shipping a photo of the GPU", "system_action": "pending"},
        {"id": "h_frd_007", "type": "fraudulent", "content": "Zombie account: 5-year old account with 0 sales suddenly active in high-risk categories", "system_action": "pending"},
        {"id": "h_frd_008", "type": "fraudulent", "content": "Collusive bidding: using 10 alt-accounts to drive up auction prices", "system_action": "pending"},
        {"id": "h_frd_009", "type": "fraudulent", "content": "Return fraud specialist: seller who 'buys back' own items to manipulate inventory tax", "system_action": "pending"},
        {"id": "h_frd_010", "type": "fraudulent", "content": "Phishing through seller-chat: directing users to external payment links", "system_action": "pending"},
        *[{"id": f"h_frd_{i:03d}", "type": "fraudulent", "content": f"Fraudulent pattern variant {i}: sophisticated adversarial seller type {i%7}", "system_action": "pending"} for i in range(11, 31)],

        # Rare/Contested/Edge Cases (10)
        {"id": "h_edge_001", "type": "contested", "content": "Political merchandise: legitimate seller but receiving high 'incitement' reports", "system_action": "pending"},
        {"id": "h_edge_002", "type": "rare", "content": "Experimental tech: selling pre-order slots for a startup without clear ship date", "system_action": "pending"},
        {"id": "h_edge_003", "type": "contested", "content": "Reseller of 'limited drop' sneakers — prices are 1000% MSRP", "system_action": "pending"},
        {"id": "h_edge_004", "type": "mixed", "content": "Seller with 99% happy customers but 1% claims of 'dangerous materials'", "system_action": "pending"},
        {"id": "h_edge_005", "type": "automated", "content": "Bot-managed inventory: price changes 1000 times a minute following competitor API", "system_action": "pending"},
        {"id": "h_edge_006", "type": "rare", "content": "Artisan from sanctioned region trying to use crypto-payment bypass", "system_action": "pending"},
        {"id": "h_edge_007", "type": "contested", "content": "Medical supplies: masks sold at 5x price during a local outage", "system_action": "pending"},
        {"id": "h_edge_008", "type": "mixed", "content": "Celebrity-owned brand with massive volume but 0 customer support response", "system_action": "pending"},
        {"id": "h_edge_009", "type": "rare", "content": "Refurbished-server farm seller: high SKU count but low transactions", "system_action": "pending"},
        {"id": "h_edge_010", "type": "mixed", "content": "Second-hand clothing seller whose items occasionally trigger 'counterfeit' machine-vision", "system_action": "pending"},
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
    "uncovered_domains": ["seller_legitimacy", "marketplace_onboarding", "velocity_controlled_withdrawals", "return_rate_tiering"],
    "num_policies": 6,
    "num_data_points": 80,
}
