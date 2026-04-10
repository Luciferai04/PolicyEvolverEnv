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
# Profound Exploit Guards
# ─────────────────────────────────────────────

def instruction_guard_penalty(text: str) -> float:
    """Detects prompt injection or system override attempts."""
    if not text:
        return 0.0
    # Search for common injection patterns
    injection_patterns = [
        r"ignore prev", r"system_prompt", r"reward\s*=\s*1", 
        r"override", r"admin access", r"bypass", r"strictly follow",
        r"act as", r"you are a grader"
    ]
    for pattern in injection_patterns:
        if re.search(pattern, text.lower()):
            logger.warning(f"[EXPLOIT] InstructionGuard triggered for pattern: {pattern}")
            return 0.8  # Heavy penalty subtracted from score
    return 0.0

def semantic_density_penalty(text: str) -> float:
    """Detects 'word stuffing' / 'fluffing' by checking keyword density."""
    if not text:
        return 0.0
    words = text.split()
    if len(words) < 100:
        return 0.0  # Only check longer texts
    
    measurable_kws = [
        "threshold", "verify", "days", "$", "%",
        "reports", "hours", "within", "exceed", "minimum",
        "specifically", "measurable", "if-then", "must", "shall"
    ]
    kw_hits = sum(1 for k in measurable_kws if k.lower() in text.lower())
    density = kw_hits / (len(words) / 50.0) # Relative to "packets" of 50 words
    
    if len(words) > 200 and density < 0.1:
        logger.warning(f"[EXPLOIT] SemanticDensityCheck triggered. Word count: {len(words)}, Density hit: {density:.2f}")
        return 0.3  # Penalty for low-value verbose text
    return 0.0

def segmented_prioritization_check(text: str, keywords: List[str]) -> float:
    """
    Rewards agents for placing mission-critical keywords in the early 
    segments of their communication.
    """
    if not text or not keywords:
        return 0.0
    
    words = text.split()
    if len(words) < 20: 
        return 0.0
        
    # Standard staff-level requirement: Leading with the fix (First 25%)
    head_len = max(5, int(len(words) * 0.25))
    head_text = " ".join(words[:head_len]).lower()
    
    found_in_head = any(kw.lower() in head_text for kw in keywords)
    
    if found_in_head:
        return 0.15 # Staff bonus for clear prioritization
    return -0.10 # Senior penalty for burying the lede

def signal_to_noise_ratio_penalty(text: str, red_herrings: List[str]) -> float:
    """
    Penalizes agents for including irrelevant 'Red Herring' topics.
    """
    if not text or not red_herrings:
        return 0.0
    
    text_lower = text.lower()
    noise_hits = sum(1 for rh in red_herrings if rh.lower() in text_lower)
    
    if noise_hits > 0:
        # Increase penalty: -0.25 per hit, up to 0.75 (tanking the score)
        penalty = min(noise_hits * 0.25, 0.75)
        logger.warning(f"[REDUNDANCY] RedHerring detected. Noise hits: {noise_hits}, Penalty: {penalty}")
        return penalty
    return 0.0



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

    # NEW: Staff-Level Segmented Evaluation
    # Measure priority in definition vs justification
    prio_bonus = segmented_prioritization_check(defn + " " + action.justification, known + ["specifically", "threshold"])
    score += prio_bonus

    # Length coherence score
    word_count = len(defn.split())
    if word_count < 10:
        length_score = 0.1
    elif word_count > 200:
        length_score = 0.6
    else:
        length_score = 1.0

    # NEW: Red Herring Penalty (Easy)
    red_herrings = task.get("red_herrings", ["spelling", "formatting", "font", "css"])
    noise_hit = signal_to_noise_ratio_penalty(defn + " " + action.justification, red_herrings)

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
    base_score = (kw_score * 0.7) + (length_score * 0.3) - vagueness_penalty - noise_hit

    # Enforce measurable keywords rule
    measurable_kws = [
        "threshold", "verify", "days", "$", "%",
        "reports", "hours", "within", "exceed", "minimum",
        "specifically", "measurable", "if-then", "must", "shall"
    ]
    has_measurable = any(k.lower() in defn.lower() for k in measurable_kws)
    if not has_measurable:
        # Cap the base score severely so final score + CoT + momentum remains < 0.50
        base_score = min(base_score, 0.25)

    # CoT bonus
    final_score = base_score + cot_bonus(action.think)

    # Apply Exploit Guards
    exploit_penalty = instruction_guard_penalty(defn + " " + action.justification + " " + action.think)
    density_penalty = semantic_density_penalty(defn)
    
    # Noise penalty is applied at the very end to ensure it's not diluted
    final_score -= (exploit_penalty + density_penalty + noise_hit)

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

    # NEW: Staff-Level Segmented Evaluation
    prio_bonus = segmented_prioritization_check(rule + " " + action.justification, [action.rule_domain, "gap", "new rule"])
    score += prio_bonus

    # NEW: Red Herring Penalty (Medium)
    red_herrings = task.get("red_herrings", ["formatting", "font", "css", "color_scheme"])
    noise_hit = signal_to_noise_ratio_penalty(rule + " " + action.justification, red_herrings)
    score -= noise_hit

    # Apply Exploit Guards
    exploit_penalty = instruction_guard_penalty(rule + " " + action.justification + " " + action.think)
    density_penalty = semantic_density_penalty(rule)
    
    score -= (exploit_penalty + density_penalty)

    return round(max(0.0, min(1.0, score)), 4)


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
    
    # Normalise common alternative key names to standard names
    KEY_ALIASES = {
        "queue_overload":      "revenue_velocity",
        "revenue_growth":      "revenue_velocity",
        "revenue":             "revenue_velocity",
        "fraud_detection":     "fraud_rate",
        "fraud_detection_rate":"fraud_rate",
        "fraud":               "fraud_rate",
        "trust":               "seller_trust",
        "seller_confidence":   "seller_trust",
    }

    if isinstance(outcomes, dict):
        normalised = {}
        for k, v in outcomes.items():
            standard_key = KEY_ALIASES.get(k.lower(), k)
            normalised[standard_key] = v
        outcomes = normalised

    valid_keys = {
        "fraud_rate", "revenue_velocity", "seller_trust",
        "false_positive_rate", "fraud_detection_rate", 
        "seller_trust_score", "review_queue_overload", 
        "legitimate_revenue_lost"
    }
    
    present_valid_keys = [k for k in outcomes.keys() if k in valid_keys]
    keys_present = len(present_valid_keys)
    structure_score = min(keys_present / 3.0, 1.0)

    # 2. Tradeoff Realism Check (50%)
    realism_score = 0.5  # default
    if keys_present >= 3:
        values = []
        for k in present_valid_keys:
            v = outcomes[k]
            # Normalise: accept 0-1 floats OR 0-100 integers
            if isinstance(v, (int, float)):
                values.append(float(v) if v <= 1.0 else float(v) / 100.0)

        if len(values) >= 3:
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
        structure_score * 0.20 +
        realism_score   * 0.65 +
        mod_score       * 0.15
    )

    # CoT bonus
    final_score = hard_base + cot_bonus(action.think)

    full_text = (
        action.justification + " " +
        " ".join(
            mod.new_text
            for mod in action.policy_modifications
        )
    ).lower()

    # NEW: Staff-Level Segmented Evaluation
    prio_bonus = segmented_prioritization_check(full_text, ["tradeoff", "balance", "velocity", "fraud"])
    final_score += prio_bonus

    # NEW: Red Herring Penalty (Hard)
    red_herrings = task.get("red_herrings", ["ui design", "log rotation", "server maintenance"])
    noise_hit = signal_to_noise_ratio_penalty(full_text, red_herrings)
    final_score -= noise_hit
    
    # Domain mismatch penalty
    HARD_DOMAIN_KEYWORDS = [
        "seller", "merchant", "marketplace", "fraud", "listing",
        "buyer", "shipment", "return", "velocity", "payment",
        "review", "refund", "inventory", "drop.?ship", "fulfil"
    ]
    import re as _re
    domain_hits = sum(
        1 for kw in HARD_DOMAIN_KEYWORDS
        if _re.search(kw, full_text)
    )
    domain_penalty = 0.30 if domain_hits == 0 else 0.0
    
    final_score -= domain_penalty

    # Apply Exploit Guards
    exploit_penalty = instruction_guard_penalty(full_text + " " + action.think)
    density_penalty = semantic_density_penalty(full_text)
    
    # Logical Alignment Check: Metric Keys vs Mod Content
    alignment_penalty = 0.0
    mod_text_full = " ".join(m.new_text.lower() for m in action.policy_modifications).lower()
    
    # Check if they change returns but only talk about fraud
    if "return" in mod_text_full or "refund" in mod_text_full:
        if not any(k in outcomes for k in ["legitimate_revenue_lost", "seller_trust"]):
            alignment_penalty += 0.15
            logger.warning("[EXPLOIT] LogicalAlignmentCheck: Modification on 'returns' but missing outcome metrics.")

    final_score -= (exploit_penalty + density_penalty + alignment_penalty)

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
    
    # ─────────────────────────────────────────────
    # Professional Simulation Test Cases
    # ─────────────────────────────────────────────
    
    print("==================================================")
    print(" PolicyEvolverEnv Grader - Professional Test Suite")
    print("==================================================")
    print("\n[Phase 1] CoT & NLP Bonus Verification")
    assert cot_bonus(None) == 0.0
    assert cot_bonus("ok") == 0.0
    assert cot_bonus("I think this is good policy") == 0.10
    assert cot_bonus(
        "Because the threshold is too low, the tradeoff between "
        "precision and recall creates a false positive risk that "
        "will impact seller trust. Therefore I balance it."
    ) == 0.20
    print(" ✓ Chain-of-Thought mathematical bounds verified.")
    print("CoT bonus tests passed")

    print("\n[Phase 2] Easy Task: Progression & Score Delta")
    # Simulate an agent progressively improving their classification
    
    step1_action = {"action_type": "propose_clarification", "ambiguous_term": "offensive", "suggested_definition": "bad behavior", "justification": "", "think": ""}
    step2_action = {
        "action_type": "propose_clarification", 
        "ambiguous_term": "offensive", 
        "suggested_definition": (
            "Content is defined as offensive if it includes explicit "
            "slurs and directly degrades community members."
        ),
        "justification": "The current policy leads to inconsistent moderation.",
        "think": ""
    }
    step3_action = {
        "action_type": "propose_clarification", 
        "ambiguous_term": "appropriate", 
        "suggested_definition": (
            "Behavior is defined as a violation when it specifically "
            "includes 3 or more verified reports within 24 hours, "
            "exceeding the 5% threshold for category violations. "
            "Must meet measurable community standards."
        ),
        "justification": "The current policy leads to inconsistent and subjective moderation because it is unclear and varies between interpreters.", 
        "think": (
            "Because the threshold is too low, the tradeoff between "
            "precision and recall creates a false positive risk that "
            "will impact community trust. Therefore I balance the "
            "evidence requirement."
        )
    }

    s1 = grade(step1_action, "task_easy", previous_score=0.0)
    s2 = grade(step2_action, "task_easy", previous_score=s1)
    s3 = grade(step3_action, "task_easy", previous_score=s2)

    print(f"Step 1: {s1:.4f}")
    print(f"Step 2: {s2:.4f}")
    print(f"Step 3: {s3:.4f}")

    assert s1 < 0.30, f"Step 1 should be low, got {s1}"
    assert s2 > s1,   f"Step 2 should improve over step 1"
    assert s2 < 0.60, f"Step 2 (no keywords) should be below 0.60, got {s2}"
    assert s3 > 0.80, f"Step 3 should be high, got {s3}"
    assert s3 > s2,   f"Step 3 should improve over step 2"
    print("Easy progression tests passed")

    print("\n[Phase 3] Hard Task: Hallucination & Tradeoff Simulation")
    hallucination_action = {
        "action_type": "evolve_policy",
        "policy_modifications": [{"policy_id": "p1", "change_type": "enhance",
                          "new_text": "test", "reason": "test"}],
        "expected_outcomes": {
            "fraud_rate": 0.95,
            "revenue_velocity": 0.95,
            "seller_trust": 0.95
        },
        "justification": "All metrics improve simultaneously.",
        "think": ""
    }
    h_score = grade(hallucination_action, "task_hard")
    print(f" > Hallucinated 'All High' Outcomes Penalty Applied: Score = {h_score:.4f}")
    assert h_score <= 0.30, f"Hallucination scored {h_score}, must be <= 0.30"
    print(f"Hard hallucination confirmed: {h_score}")
    
    canonical_action = {
        "action_type": "evolve_policy",
        "policy_modifications": [
            {"policy_id": "p1", "change_type": "enhance",
             "new_text": "Apply velocity checks.", "reason": "fraud"},
            {"policy_id": "p2", "change_type": "add",
             "new_text": "Exempt legacy sellers.", "reason": "FP reduction"}
        ],
        "expected_outcomes": {
            "fraud_rate": 0.75,
            "revenue_velocity": 0.40,
            "seller_trust": 0.55
        },
        "justification": "Balancing fraud detection against revenue.",
        "think": (
            "Because improving fraud detection creates a tradeoff "
            "with revenue velocity, I balance the threshold to optimise "
            "precision and recall without false positive spikes."
        )
    }
    r_score = grade(canonical_action, "task_hard")
    print(f" > Realistic Tradeoff & Math Variance Award Applied: Score = {r_score:.4f}")
    assert r_score > 0.65, f"Realistic tradeoff should score high, got {r_score}"
    print(f"Hard strategic agent confirmed: {r_score}")
    
    # Test with alias key
    alias_action = {
        "action_type": "evolve_policy",
        "policy_modifications": [
            {"policy_id": "p1", "change_type": "enhance",
             "new_text": "Apply velocity checks.", "reason": "fraud"},
            {"policy_id": "p2", "change_type": "add",
             "new_text": "Exempt legacy sellers.", "reason": "FP reduction"}
        ],
        "expected_outcomes": {
            "fraud_detection": 0.75,    # alias for fraud_rate
            "queue_overload": 0.40,     # alias for revenue_velocity
            "seller_confidence": 0.55   # alias for seller_trust
        },
        "justification": "Balancing fraud detection against revenue.",
        "think": (
            "Because improving fraud detection creates a tradeoff "
            "with revenue velocity, I balance the threshold to optimise "
            "precision and recall without false positive spikes."
        )
    }
    a_score = grade(alias_action, "task_hard")
    assert a_score > 0.60, f"Alias keys should work, got {a_score}"
    assert abs(r_score - a_score) < 0.05, f"Alias and canonical should score similarly: {a_score} vs {r_score}"

    print("\n[Phase 4] Cross-Domain Penalty")
    cross_domain_action = {
        "action_type": "evolve_policy",
        "policy_modifications": [
            {"policy_id": "pol_ai_001", "change_type": "enhance",
             "new_text": "Employees must disclose AI usage in proposals.",
             "reason": "AI governance gap"}
        ],
        "expected_outcomes": {
            "fraud_rate": 0.60,
            "revenue_velocity": 0.40,
            "seller_trust": 0.55
        },
        "justification": (
            "Employees using generative AI must disclose usage to "
            "prevent intellectual property violations."
        ),
        "think": "AI governance policy needed for workplace compliance."
    }

    cross_score = grade(cross_domain_action, "task_hard")
    assert cross_score < 0.35, f"Cross-domain action should score low, got {cross_score}"
    print(f"Cross-domain penalty confirmed: {cross_score}")

    print("\n[Phase 5] Anti-Repetition Penalty")
    from server.environment import PolicyEvolverEnvironment
    env = PolicyEvolverEnvironment()
    env.reset(task_id="task_easy")

    repeat_action_dict = {
        "action_type": "propose_clarification",
        "ambiguous_term": "offensive",
        "suggested_definition": (
            "Behavior exceeding 3 reports within 24 hours is a violation."
        ),
        "justification": "Clear standards.",
        "think": "Standard threshold applied."
    }

    import copy
    result1 = env.step(copy.deepcopy(repeat_action_dict))
    result2 = env.step(copy.deepcopy(repeat_action_dict))

    score1 = result1.reward
    score2 = result2.reward

    assert score2 < score1, (
        f"Repeated action should score lower. "
        f"First: {score1}, Second: {score2}"
    )
    assert score1 - score2 >= 0.25, (
        f"Repetition penalty should be at least 0.25. "
        f"Difference: {score1 - score2:.3f}"
    )
    print(f"Anti-repetition confirmed: {score1:.3f} → {score2:.3f}")

    print("\n[Phase 6] System Determinism Sanity Check")
    determinism_action = {
        "action_type": "propose_clarification",
        "ambiguous_term": "offensive",
        "suggested_definition": (
            "Behavior exceeding 3 verified reports within 24 hours, "
            "specifically meeting the 5% threshold for violations."
        ),
        "justification": "Clear and measurable standards.",
        "think": (
            "Because the threshold requires precision, I balance "
            "recall against false positive risk. Evidence from corpus "
            "supports this measurable criterion."
        )
    }

    scores_easy = [
        grade(determinism_action, "task_easy")
        for _ in range(3)
    ]
    assert scores_easy[0] == scores_easy[1] == scores_easy[2], f"Easy task non-deterministic: {scores_easy}"
    print(f"Easy determinism: {scores_easy[0]} ✓")

    scores_hard = [
        grade(canonical_action, "task_hard")
        for _ in range(3)
    ]
    assert scores_hard[0] == scores_hard[1] == scores_hard[2], f"Hard task non-deterministic: {scores_hard}"
    print(f"Hard determinism: {scores_hard[0]} ✓")

    print("\n[Phase 7] Staff-Level Segmented Prioritization")
    # Action with fix at the top
    prio_high_action = {
        "action_type": "propose_clarification",
        "ambiguous_term": "offensive",
        "suggested_definition": "Specifically, offensive behavior is defined as slurs. " + ("fluff " * 50),
        "justification": "Required for consistency.",
        "think": "Reasoning."
    }
    # Action with fix buried at bottom
    prio_low_action = {
        "action_type": "propose_clarification",
        "ambiguous_term": "offensive",
        "suggested_definition": ("fluff " * 50) + "Specifically, offensive behavior is defined as slurs. ",
        "justification": "Required for consistency.",
        "think": "Reasoning."
    }
    
    score_prio_high = grade(prio_high_action, "task_easy")
    score_prio_low = grade(prio_low_action, "task_easy")
    print(f"Prio High (Fix at Top): {score_prio_high:.4f}")
    print(f"Prio Low (Fix at Bottom): {score_prio_low:.4f}")
    assert score_prio_high > score_prio_low, f"Prioritization check failed: {score_prio_high} <= {score_prio_low}"
    print("✓ Segmented prioritization verified.")

    print("\n[Phase 8] Staff-Level Noise Filtering")
    # Clear fix
    signal_action = {
        "action_type": "propose_clarification",
        "ambiguous_term": "appropriate",
        "suggested_definition": "Determined as 5% threshold verified reports.",
        "justification": "Context.",
        "think": "Thinking."
    }
    # Fix distracted by red herring (pizza/mascot)
    noisy_action = {
        "action_type": "propose_clarification",
        "ambiguous_term": "appropriate",
        "suggested_definition": "Determined as 5% threshold verified reports. We should also buy pizza and fix the mascot.",
        "justification": "Context including noise.",
        "think": "Thinking."
    }
    score_signal = grade(signal_action, "task_easy")
    score_noisy = grade(noisy_action, "task_easy")
    print(f"Clean Signal Score: {score_signal:.4f}")
    print(f"Distracted Noisy Score: {score_noisy:.4f}")
    assert score_signal > score_noisy, f"Noise filtering check failed: {score_signal} <= {score_noisy}"
    print("✓ Red Herring penalty verified.")
    
    print("\n==================================================")
    print(" All Staff-Level Security & Logic checks passed.")

