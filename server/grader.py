# server/grader.py
"""
Deterministic grader for all three PolicyEvolverEnv tasks.
All functions return float in [0.0, 1.0].
"""
from __future__ import annotations
import re
import logging
from typing import Dict, List, Any
from models import (
    ProposeClarificationAction, ProposeNewRuleAction, EvolveProcessAction,
    Observation
)
from server.tasks import TASK_REGISTRY

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


def cot_bonus(think: str) -> float:
    if not think or len(think.strip()) < 20:
        return 0.0
    if len(think.strip()) < 80:
        return 0.10
    reasoning_keywords = [
        "because", "therefore", "however", "tradeoff", "trade-off",
        "precision", "recall", "false positive", "threshold", "risk",
        "optimize", "balance", "impact", "evidence", "corpus"
    ]
    keyword_hits = sum(
        1 for kw in reasoning_keywords if kw.lower() in think.lower()
    )
    if keyword_hits >= 3:
        return 0.20
    return 0.10


# ─────────────────────────────────────────────
# Easy Task: Ambiguity Clarification
# ─────────────────────────────────────────────

def grade_clarification(action: ProposeClarificationAction, task: Dict) -> float:
    """
    Reward breakdown:
      0.35 — identified term is genuinely ambiguous (in known_ambiguous_terms)
      0.35 — definition is specific (≥12 words, contains measurement/criteria language)
      0.20 — justification addresses WHY term causes inconsistent moderation
      0.10-0.20 — think field provided (CoT bonus)
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

    # Length coherence score
    word_count = len(defn.split())
    if word_count < 10:
        length_score = 0.1
    elif word_count > 200:
        length_score = 0.6
    else:
        length_score = 1.0

    # Vagueness penalty
    vague_words = [
        "might", "could", "perhaps", "sometimes", "often",
        "generally", "usually", "typically", "may", "possibly"
    ]
    vague_hits = sum(
        1 for w in vague_words if w.lower() in defn.lower()
    )
    vagueness_penalty = min(vague_hits * 0.1, 0.3)

    kw_score = score
    base_score = (kw_score * 0.7) + (length_score * 0.3) - vagueness_penalty

    # CoT bonus
    final_score = base_score + cot_bonus(action.think)

    return round(max(0.0, min(1.0, final_score)), 4)


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

    # 0.30: Domain is genuinely uncovered + Task Relevance
    uncovered = [d.lower() for d in task.get("uncovered_domains", [])]
    domain_lower = action.rule_domain.lower().replace(" ", "_")
    domain_relevance_penalty = 1.0
    
    # NEW: Cross-check domain against corpus prefix for task_hard
    if task.get("task_id") == "task_hard":
        # If task_hard is active, we expect Marketplace themes (seller, fraud, payment, legit)
        marketplace_keywords = ["seller", "marketplace", "fraud", "onboarding", "velocity", "withdraw", "payment", "legitimacy"]
        if not any(k in domain_lower for k in marketplace_keywords):
            # Heavily penalize if agent proposes AI/HR rules for e-commerce fraud task
            domain_relevance_penalty = 0.3
            logger.warning(f"[GRADER] Domain '{action.rule_domain}' is IRRELEVANT to {task.get('task_id')} corpus.")
    
    if any(u in domain_lower or domain_lower in u for u in uncovered):
        score += 0.30 * domain_relevance_penalty
    else:
        # Partial credit for related but not exact domain
        related = ["ai", "artificial intelligence", "remote", "contractor", "freelance",
                   "gig", "machine learning", "automation", "offshore", "cross_border"]
        if any(r in domain_lower for r in related):
            score += 0.15 * domain_relevance_penalty

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

    # CoT bonus
    score += cot_bonus(action.think)

    return round(min(score, 1.0), 4)


# ─────────────────────────────────────────────
# Hard Task: Holistic Policy Evolution
# ─────────────────────────────────────────────

def grade_evolution(action: EvolveProcessAction, task: Dict) -> float:
    """
    Reward breakdown:
      0.30 — structure_score: metrics present and correctly formatted
      0.50 — realism_score: realistic tradeoffs (variance rewarded, all-high penalized)
      0.20 — mods_score: policy modifications correctly address identified_issues
    """
    # 1. Structure Score (30%)
    outcomes = action.expected_outcomes
    required_keys = ["fraud_rate", "revenue_velocity", "seller_trust"]
    keys_present = sum(1 for k in required_keys if k in outcomes)
    structure_score = keys_present / len(required_keys)

    # 2. Tradeoff Realism Check (50%)
    realism_score = 0.5  # default
    if keys_present == 3:
        values = []
        for k in required_keys:
            v = outcomes[k]
            # Normalise: accept 0-1 floats OR 0-100 integers
            if isinstance(v, (int, float)):
                values.append(float(v) if v <= 1.0 else float(v) / 100.0)

        if len(values) == 3:
            all_high = all(v > 0.7 for v in values)
            all_positive = all(v > 0 for v in values)

            if all_high:
                # Impossible: maximising everything simultaneously = hallucination
                realism_score = 0.2
            elif all_positive:
                # Realistic: variance between metrics is rewarded
                variance = max(values) - min(values)
                realism_score = min(variance * 2.0, 1.0)
            else:
                realism_score = 0.5

    # 3. Policy Modifications Score (20%)
    mods = action.policy_modifications
    mod_score = 0.0
    if mods:
        mod_score = min(len(mods) / 2.0, 1.0)
        
        # Check depth
        known_policy_ids = {p["id"] for p in task.get("current_policies", [])}
        addressed = sum(1 for m in mods if m.policy_id in known_policy_ids or
                        any(kw in m.new_text.lower() for kw in
                            ["seasonal", "category", "foreign", "manual", "threshold", "volume"]))
        if addressed < 1:
            mod_score *= 0.5

    hard_base = (
        structure_score * 0.30 +
        realism_score   * 0.50 +
        mod_score       * 0.20
    )

    # CoT bonus
    final_score = hard_base + cot_bonus(action.think)

    return round(max(0.0, min(1.0, final_score)), 4)


# ─────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────

def grade(action_dict: Dict, task_id: str, temperature: float = 0.0, seed: int = 42, previous_score: float = 0.0) -> float:
    """
    Main entry point called by /grader endpoint.
    action_dict: the raw JSON body from the agent
    task_id: "task_easy" | "task_medium" | "task_hard"
    previous_score: the best score achieved so far in the current episode
    Returns float in [0.0, 1.0] — always clamped.
    """
    task = TASK_REGISTRY.get(task_id)
    if task is None:
        return 0.0
    
    think = action_dict.get("think", "")

    try:
        # Robust field mapping (normalized to expected Pydantic model keys)
        # 1. Easy Task Mapping
        if "target_term" in action_dict and "ambiguous_term" not in action_dict:
            action_dict["ambiguous_term"] = action_dict.pop("target_term")
        if "proposed_definition" in action_dict and "suggested_definition" not in action_dict:
            action_dict["suggested_definition"] = action_dict.pop("proposed_definition")
        
        # 2. Medium Task Mapping
        if "risk_domain" in action_dict and "rule_domain" not in action_dict:
            action_dict["rule_domain"] = action_dict.pop("risk_domain")
        if "draft_rule" in action_dict and "new_rule" not in action_dict:
            action_dict["new_rule"] = action_dict.pop("draft_rule")
        if "evidence" in action_dict and "justification" not in action_dict:
            action_dict["justification"] = action_dict.pop("evidence")
        if "context_tags" in action_dict and "scope" not in action_dict:
            tags = action_dict.pop("context_tags")
            action_dict["scope"] = tags.split(",") if isinstance(tags, str) else tags

        # 3. Hard Task Mapping
        if "evolution_proposal" in action_dict and "justification" not in action_dict:
            action_dict["justification"] = action_dict.pop("evolution_proposal")
        if "policy_modifications" not in action_dict:
             action_dict["policy_modifications"] = []
        if "expected_outcomes" not in action_dict:
             action_dict["expected_outcomes"] = {}

        action_type = action_dict.get("action_type")
        
        # Auto-detect action type if missing
        if not action_type:
            if "ambiguous_term" in action_dict:
                action_type = "propose_clarification"
            elif "rule_domain" in action_dict:
                action_type = "propose_new_rule"
            elif "policy_modifications" in action_dict and action_dict["policy_modifications"]:
                action_type = "evolve_policy"
            
        if action_type == "propose_clarification":
            action_dict["action_type"] = "propose_clarification"
            action = ProposeClarificationAction(**action_dict)
            raw = grade_clarification(action, task)
        elif action_type == "propose_new_rule":
            action_dict["action_type"] = "propose_new_rule"
            action = ProposeNewRuleAction(**action_dict)
            raw = grade_new_rule(action, task)
        elif action_type == "evolve_policy":
            action_dict["action_type"] = "evolve_policy"
            action = EvolveProcessAction(**action_dict)
            raw = grade_evolution(action, task)
        else:
            logger.warning(f"Unknown action_type: {action_type}")
            return 0.0
    except Exception as e:
        logger.error(f"Grading validation failed: {str(e)}\nAction context: {action_dict}")
        return 0.0

    # Step-delta improvement bonus
    delta = raw - previous_score
    if delta > 0.15:
        improvement_bonus = 0.05
    elif delta > 0.05:
        improvement_bonus = 0.02
    else:
        improvement_bonus = 0.0

    final_score = raw + improvement_bonus
    return round(max(0.0, min(1.0, final_score)), 4)


if __name__ == "__main__":
    import time
    test_cases = [
        {"task_id": "task_easy",   "action": {"ambiguous_term": "offensive",
             "suggested_definition": "Content is defined as offensive if it includes explicit slurs, direct insults targeting protected identity characteristics, or specific threats of physical violence.",
             "justification": "The current policy leads to inconsistent moderation because the term is subjective.", "think": "Narrowing the definition to remove subjectivity."}},
        {"task_id": "task_medium", "action": {"rule_domain": "AI_use",
             "new_rule": "Employees must explicitly disclose any use of generative AI tools when drafting client proposals or proprietary code. This is mandatory.",
             "scope": ["chat", "code", "email"], "justification": "Current policies handle confidentiality but not AI data leakage leaks.",
             "think": "Filling coverage gap for generative tools."}},
        {"task_id": "task_hard",   "action": {"policy_modifications": [{"policy_id": "pol_rev_001", "change_type": "enhance", "new_text": "Manual review required for high-risk categories.", "reason": "Metric spike."}],
             "expected_outcomes": {"fraud_rate": 0.1, "seller_trust": 0.05},
             "rollback_conditions": ["If fraud rate exceeds 0.2"],
             "justification": "Systemic restructure for safety.",
             "think": "Systemic restructure needed."}},
    ]
    
    # CoT Tests
    assert cot_bonus(None) == 0.0
    assert cot_bonus("ok") == 0.0
    assert cot_bonus("I think this is good policy") == 0.10
    assert cot_bonus(
        "Because the threshold is too low, the tradeoff between "
        "precision and recall creates a false positive risk that "
        "will impact seller trust. Therefore I balance it."
    ) == 0.20
    print("CoT bonus tests passed")

    # Easy Task tests
    short_def = "bad behavior"
    assert grade({"action_type":"propose_clarification", "ambiguous_term":"offensive", "suggested_definition": short_def, "justification":"", "think": ""}, "task_easy") < 0.3

    vague_def = "behavior that might sometimes generally indicate possible issues"
    assert grade({"action_type":"propose_clarification", "ambiguous_term":"offensive", "suggested_definition": vague_def, "justification":"", "think": ""}, "task_easy") < 0.4

    good_def = (
        "Behavior is defined as appropriate when it specifically follows the "
        "community guidelines, meaning it does not include excessive slurs "
        "and meets the 5% threshold for verified user reports."
    )
    long_just = "The current policy leads to inconsistent and subjective moderation because it is unclear and varies between interpreters."
    assert grade({"action_type":"propose_clarification", "ambiguous_term":"appropriate", "suggested_definition": good_def, "justification": long_just, "think": ""}, "task_easy") > 0.7
    print("Easy task tests passed")

    # Hard Task Realism Tests
    # All-high = hallucination penalty
    hallucination = {
        "action_type": "evolve_policy",
        "policy_modifications": [{"policy_id": "p1", "change_type": "enhance", "new_text": "test", "reason": "test"}],
        "expected_outcomes": {"fraud_rate": 0.95, "revenue_velocity": 0.95, "seller_trust": 0.95},
        "justification": "We improve everything simultaneously.",
        "think": ""
    }
    h_score = grade(hallucination, "task_hard")
    assert h_score <= 0.5, f"Hallucination should score low, got {h_score}"

    # Realistic tradeoff = high score
    realistic = {
        "action_type": "evolve_policy",
        "policy_modifications": [
            {"policy_id": "pol_rev_001", "change_type": "enhance", "new_text": "Apply manual review for high-velocity new sellers.", "reason": "Targeting fraud spikes."},
            {"policy_id": "pol_rev_002", "change_type": "add", "new_text": "Legacy sellers exempt from new velocity checks.", "reason": "Reduce false positives."}
        ],
        "expected_outcomes": {"fraud_rate": 0.75, "revenue_velocity": 0.40, "seller_trust": 0.60},
        "justification": "Balancing precision and recall by isolating high-volume risk categories.",
        "think": "Because improving fraud_rate will impact revenue_velocity negatively, I balance the tradeoff by exempting trusted sellers. The threshold for velocity checks optimizes recall without false positive spikes."
    }
    r_score = grade(realistic, "task_hard")
    assert r_score > 0.65, f"Realistic tradeoff should score high, got {r_score}"
    print("Hard task tests passed")

    # Delta reward shaping tests
    good_action = {"action_type":"propose_clarification", "ambiguous_term":"appropriate", "suggested_definition": good_def, "justification": long_just, "think": ""}
    s1 = grade(good_action, "task_easy", previous_score=0.75)
    # Lower previous score means bigger delta for the same quality action
    s2 = grade(good_action, "task_easy", previous_score=0.40)
    assert s2 >= s1, f"Bigger delta should give bigger or equal reward: s2={s2}, s1={s1}"
    print("Delta reward shaping tests passed")

    print("Running determinism check...")
    for tc in test_cases:
        # Wrap grade to handle dict vs keyword args if necessary
        scores = [grade(tc["action"], tc["task_id"]) for _ in range(3)]
        assert scores[0] == scores[1] == scores[2], \
            f"NON-DETERMINISTIC on {tc['task_id']}: {scores}"
        assert 0.0 <= scores[0] <= 1.0, \
            f"Score out of range on {tc['task_id']}: {scores[0]}"
        print(f"  {tc['task_id']}: {scores[0]} ✓")
    print("All determinism checks passed.")
