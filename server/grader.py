# server/grader.py
"""
Deterministic grader for all three PolicyEvolverEnv tasks.
All functions return float in [0.0, 1.0].
"""
from __future__ import annotations
import re
from typing import Dict, List, Any
from ..models import (
    ProposeClarificationAction, ProposeNewRuleAction, EvolveProcessAction,
    Observation
)
from .tasks import TASK_REGISTRY


# ─────────────────────────────────────────────
# Easy Task: Ambiguity Clarification
# ─────────────────────────────────────────────

def grade_clarification(action: ProposeClarificationAction, task: Dict) -> float:
    """
    Reward breakdown:
      0.35 — identified term is genuinely ambiguous (in known_ambiguous_terms)
      0.35 — definition is specific (≥12 words, contains measurement/criteria language)
      0.20 — justification addresses WHY term causes inconsistent moderation
      0.10 — think field provided (CoT bonus)
    """
    score = 0.0

    # 0.35: Is the identified term actually ambiguous?
    known = [t.lower() for t in task.get("known_ambiguous_terms", [])]
    if action.ambiguous_term.lower() in known:
        score += 0.35
    else:
        # Partial credit if it's a word that plausibly causes ambiguity
        vague_words = ["reasonable", "substantial", "appropriate", "excessive", "significant",
                       "severe", "abusive", "hostile", "threatening", "offensive", "respectful"]
        if any(w in action.ambiguous_term.lower() for w in vague_words):
            score += 0.15

    # 0.35: Definition quality
    defn = action.suggested_definition
    defn_score = 0.0
    words = defn.split()
    if len(words) >= 12:
        defn_score += 0.10
    criteria_words = ["includes", "means", "refers to", "defined as", "encompasses",
                      "specifically", "measurable", "example", "such as", "e.g."]
    if any(w in defn.lower() for w in criteria_words):
        defn_score += 0.15
    action_words = ["will", "must", "shall", "is", "are", "requires"]
    if any(w in defn.lower() for w in action_words):
        defn_score += 0.10
    score += min(defn_score, 0.35)

    # 0.20: Justification quality
    just = action.justification.lower()
    just_score = 0.0
    if len(action.justification.split()) >= 10:
        just_score += 0.10
    inconsistency_words = ["inconsistent", "vary", "subjective", "unclear", "different",
                           "interpret", "misapply", "dispute", "ambiguous"]
    if any(w in just for w in inconsistency_words):
        just_score += 0.10
    score += min(just_score, 0.20)

    # 0.10: CoT bonus
    if action.think and len(action.think.strip()) > 20:
        score += 0.10

    return round(min(score, 1.0), 4)


# ─────────────────────────────────────────────
# Medium Task: Gap Detection + New Rule
# ─────────────────────────────────────────────

def grade_new_rule(action: ProposeNewRuleAction, task: Dict) -> float:
    """
    Reward breakdown:
      0.30 — rule_domain matches a genuinely uncovered domain
      0.30 — rule text is specific and actionable (not vague platitude)
      0.25 — scope covers multiple relevant scenarios
      0.05 — integration_points reference existing policies
      0.10 — think field provided (CoT bonus)
    """
    score = 0.0

    # 0.30: Domain is genuinely uncovered
    uncovered = [d.lower() for d in task.get("uncovered_domains", [])]
    domain_lower = action.rule_domain.lower().replace(" ", "_")
    if any(u in domain_lower or domain_lower in u for u in uncovered):
        score += 0.30
    else:
        # Partial credit for related but not exact domain
        related = ["ai", "artificial intelligence", "remote", "contractor", "freelance",
                   "gig", "machine learning", "automation", "offshore", "cross_border"]
        if any(r in domain_lower for r in related):
            score += 0.15

    # 0.30: Rule text quality
    rule = action.new_rule
    rule_score = 0.0
    if len(rule.split()) >= 15:
        rule_score += 0.10
    mandatory_words = ["must", "will", "shall", "required", "prohibited", "mandatory"]
    if any(w in rule.lower() for w in mandatory_words):
        rule_score += 0.10
    conditional_words = ["when", "if", "unless", "in cases where", "prior to", "before"]
    if any(w in rule.lower() for w in conditional_words):
        rule_score += 0.10
    # Penalise vague language
    vague = ["may", "should consider", "might", "perhaps", "in some cases"]
    if any(w in rule.lower() for w in vague):
        rule_score -= 0.10
    score += max(min(rule_score, 0.30), 0.0)

    # 0.25: Scope covers multiple scenario types
    if len(action.scope) >= 2:
        score += 0.15
    if len(action.scope) >= 4:
        score += 0.10

    # 0.05: Integration points reference existing policy IDs or domains
    if action.integration_points and len(action.integration_points) >= 1:
        score += 0.05

    # 0.10: CoT bonus
    if action.think and len(action.think.strip()) > 20:
        score += 0.10

    return round(min(score, 1.0), 4)


# ─────────────────────────────────────────────
# Hard Task: Holistic Policy Evolution
# ─────────────────────────────────────────────

def grade_evolution(action: EvolveProcessAction, task: Dict) -> float:
    """
    Reward breakdown:
      0.30 — ≥2 policy modifications; modifications address identified_issues
      0.25 — expected_outcomes are realistic and cover key metrics
      0.20 — rollback_conditions are specific (not generic)
      0.15 — justification addresses trade-offs (both sides)
      0.10 — think field provided (CoT bonus)
    """
    score = 0.0
    identified_issues = [i["issue"].lower() for i in task.get("identified_issues", [])]
    key_metrics = {o["metric"] for o in task.get("policy_outcomes", [])}

    # 0.30: Modifications address real problems
    mods = action.policy_modifications
    mod_score = 0.0
    if len(mods) >= 2:
        mod_score += 0.15
    # Check that at least one modification references a known policy ID or known issue
    known_policy_ids = {p["id"] for p in task.get("current_policies", [])}
    addressed = sum(1 for m in mods if m.policy_id in known_policy_ids or
                    any(kw in m.new_text.lower() for kw in
                        ["seasonal", "category", "foreign", "manual", "threshold", "volume"]))
    if addressed >= 1:
        mod_score += 0.10
    if addressed >= 2:
        mod_score += 0.05
    score += min(mod_score, 0.30)

    # 0.25: Expected outcomes realistic and cover key metrics
    outcomes = action.expected_outcomes
    outcome_score = 0.0
    covered_metrics = {m for m in outcomes if m in key_metrics}
    if len(covered_metrics) >= 2:
        outcome_score += 0.15
    # Values should be realistic deltas (not all 1.0)
    non_trivial = sum(1 for v in outcomes.values() if 0.01 <= v <= 0.60)
    if non_trivial >= 2:
        outcome_score += 0.10
    score += min(outcome_score, 0.25)

    # 0.20: Rollback conditions are specific
    rollbacks = action.rollback_conditions
    rollback_score = 0.0
    if len(rollbacks) >= 1:
        rollback_score += 0.10
    # Specific = contains a number or metric name
    specific = sum(1 for r in rollbacks if
                   re.search(r'\d+', r) or
                   any(m in r.lower() for m in ["false positive", "fraud", "trust", "revenue", "queue"]))
    if specific >= 1:
        rollback_score += 0.10
    score += min(rollback_score, 0.20)

    # 0.15: Justification addresses trade-offs
    just = action.justification.lower()
    trade_off_pairs = [
        (["precision", "accuracy", "false positive"], ["recall", "coverage", "missed"]),
        (["seller trust", "legitimate"], ["fraud", "detection"]),
        (["automation", "efficiency"], ["manual", "review"]),
    ]
    tradeoffs_found = 0
    for side_a, side_b in trade_off_pairs:
        if any(w in just for w in side_a) and any(w in just for w in side_b):
            tradeoffs_found += 1
    if tradeoffs_found >= 1:
        score += 0.10
    if tradeoffs_found >= 2:
        score += 0.05

    # 0.10: CoT bonus
    if action.think and len(action.think.strip()) > 20:
        score += 0.10

    return round(min(score, 1.0), 4)


# ─────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────

def grade(action_dict: Dict, task_id: str) -> float:
    """
    Main entry point called by /grader endpoint.
    action_dict: the raw JSON body from the agent
    task_id: "task_easy" | "task_medium" | "task_hard"
    Returns float in [0.0, 1.0] — always clamped.
    """
    task = TASK_REGISTRY.get(task_id)
    if task is None:
        return 0.0

    try:
        action_type = action_dict.get("action_type")
        if action_type == "propose_clarification":
            action = ProposeClarificationAction(**action_dict)
            raw = grade_clarification(action, task)
        elif action_type == "propose_new_rule":
            action = ProposeNewRuleAction(**action_dict)
            raw = grade_new_rule(action, task)
        elif action_type == "evolve_policy":
            action = EvolveProcessAction(**action_dict)
            raw = grade_evolution(action, task)
        else:
            return 0.0
    except Exception:
        return 0.0

    return max(0.0, min(1.0, raw))
