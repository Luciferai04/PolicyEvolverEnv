# baseline/run_baseline.py
"""
LLM-powered baseline for PolicyEvolverEnv.

Primary path:  Uses AsyncOpenAI client with OPENAI_API_KEY (or HF_TOKEN) to
               run a language model against all 3 environment tasks.
Fallback path: Rule-based hardcoded actions used when no API key is available.

Run:
    python -m policy_evolver_env.baseline.run_baseline                  # LLM baseline (needs OPENAI_API_KEY)
    python -m policy_evolver_env.baseline.run_baseline --mode rule      # Rule-based fallback
    python -m policy_evolver_env.baseline.run_baseline --output json    # JSON output

Expected scores (LLM):  easy ~0.80, medium ~0.70, hard ~0.55
Expected scores (rule): easy ~0.65, medium ~0.50, hard ~0.35

Required env vars:
    OPENAI_API_KEY   — OpenAI key or HF Inference API token (primary)
    HF_TOKEN         — Hugging Face token (fallback if no OPENAI_API_KEY)
    API_BASE_URL     — API endpoint (default: https://api.openai.com/v1)
    MODEL_NAME       — Model to use (default: meta-llama/Llama-3.3-70B-Instruct)
    OPENENV_BASE_URL — Environment server (default: http://localhost:7860)
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional

import httpx
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Configuration (all from env vars)
# ─────────────────────────────────────────────

BASE_URL = os.getenv("OPENENV_BASE_URL", "http://127.0.0.1:7860")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN", "") or os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")


def verify_environment() -> bool:
    """Verify required env vars. Returns True if LLM mode is possible."""
    if not API_KEY:
        logger.warning(
            "No API_KEY (HF_TOKEN) found. "
            "LLM baseline will be skipped. Set one of these env vars to enable it."
        )
        return False
    logger.info(f"API key found. Model: {MODEL_NAME}  Base URL: {API_BASE_URL}")
    return True


# ─────────────────────────────────────────────
# LLM Agent
# ─────────────────────────────────────────────

class PolicyEvolverAgent:
    """LLM-powered agent that calls the OpenAI-compatible API."""

    def __init__(self):
        self.client = OpenAI(
            api_key=API_KEY,
            base_url=API_BASE_URL,
        )
        self.model = MODEL_NAME

    def _call(self, prompt: str, max_tokens: int = 700, temperature: float = 0.3) -> Optional[Dict]:
        """Call the LLM and parse JSON response. Returns None on failure."""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a senior policy analyst. "
                            "Always respond with a single valid JSON object and nothing else. "
                            "No markdown fences, no preamble."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            raw = resp.choices[0].message.content.strip()
            # Strip accidental markdown fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return None

    def handle_easy(self, obs: Dict) -> Dict:
        """Easy task: propose clarification for an ambiguous policy term."""
        prompt = f"""
Analyze the following social media platform policies and user-generated data.
Identify ONE genuinely ambiguous term that causes inconsistent moderation decisions.
Propose a specific, measurable definition.

POLICIES:
{json.dumps(obs.get("current_policies", []), indent=2)}

DATA EXAMPLES (how posts were actually handled):
{json.dumps(obs.get("data_corpus", [])[:6], indent=2)}

Respond ONLY with this JSON schema:
{{
  "action_type": "propose_clarification",
  "ambiguous_term": "<the exact term from policies>",
  "suggested_definition": "<specific, ≥15 word definition with clear criteria>",
  "affected_policy_ids": ["<policy id>"],
  "justification": "<why inconsistent moderation results; ≥15 words>",
  "think": "<step-by-step reasoning: which posts were handled inconsistently and why>"
}}
"""
        result = self._call(prompt, max_tokens=600)
        if result:
            result["action_type"] = "propose_clarification"
            return result
        # Fallback
        return RULE_BASED_ACTIONS["task_easy"]

    def handle_medium(self, obs: Dict) -> Dict:
        """Medium task: detect policy gap and propose new rule."""
        prompt = f"""
You are reviewing corporate HR policies. The data shows real incidents that occurred.
Find ONE scenario category NOT adequately covered by existing policies.
Propose a specific, mandatory new rule to fill the gap.

EXISTING POLICIES:
{json.dumps(obs.get("current_policies", []), indent=2)}

INCIDENT DATA:
{json.dumps(obs.get("data_corpus", []), indent=2)}

Respond ONLY with this JSON schema:
{{
  "action_type": "propose_new_rule",
  "rule_domain": "<e.g. AI_use | gig_worker_post_engagement | cross_border_remote>",
  "new_rule": "<mandatory rule using 'must'/'shall'/'required'; ≥20 words; no vague language>",
  "scope": ["<scenario 1>", "<scenario 2>", "<scenario 3>", "<scenario 4>"],
  "integration_points": ["<existing policy id 1>", "<existing policy id 2>"],
  "justification": "<cite specific incident IDs and why gap exists; ≥20 words>",
  "think": "<which incident type appears most frequently uncovered and why a rule is needed>"
}}
"""
        result = self._call(prompt, max_tokens=800)
        if result:
            result["action_type"] = "propose_new_rule"
            return result
        return RULE_BASED_ACTIONS["task_medium"]

    def handle_hard(self, obs: Dict) -> Dict:
        """Hard task: holistic policy evolution with trade-off reasoning."""
        prompt = f"""
You are a senior Trust & Safety policy architect. The current policy framework is
underperforming. Propose specific modifications to ≥2 existing policies to improve
both precision (reduce false positives) and recall (catch more fraud) simultaneously.
Acknowledge the trade-offs explicitly.

CURRENT POLICIES:
{json.dumps(obs.get("current_policies", []), indent=2)}

PERFORMANCE METRICS (current vs target):
{json.dumps(obs.get("policy_outcomes", []), indent=2)}

SYSTEM METRICS:
{json.dumps(obs.get("system_metrics", {}), indent=2)}

KNOWN ISSUES:
{json.dumps(obs.get("identified_issues", []), indent=2)}

Respond ONLY with this JSON schema:
{{
  "action_type": "evolve_policy",
  "policy_modifications": [
    {{
      "policy_id": "<exact policy id from above>",
      "change_type": "enhance",
      "new_text": "<specific replacement text; must be context-aware, not blanket>",
      "reason": "<cite the specific metric that proves current policy fails>"
    }},
    {{
      "policy_id": "<second policy id>",
      "change_type": "enhance",
      "new_text": "<replacement text>",
      "reason": "<metric-backed reason>"
    }}
  ],
  "expected_outcomes": {{
    "false_positive_rate": <realistic delta 0.01-0.40>,
    "fraud_detection_rate": <realistic delta 0.01-0.40>,
    "seller_trust_score": <realistic delta 0.01-0.30>,
    "review_queue_overload": <realistic delta 0.01-0.40>
  }},
  "rollback_conditions": [
    "<specific numeric threshold that triggers revert>",
    "<second specific condition with metric name and number>"
  ],
  "justification": "<explain trade-offs: what improves, what worsens, and why net positive>",
  "think": "<identify the two worst-performing metrics and trace root cause to specific policy>"
}}
"""
        result = self._call(prompt, max_tokens=1200, temperature=0.2)
        if result:
            result["action_type"] = "evolve_policy"
            return result
        return RULE_BASED_ACTIONS["task_hard"]


# ─────────────────────────────────────────────
# Environment interaction helpers (HTTP-based)
# ─────────────────────────────────────────────

async def env_reset(client: httpx.AsyncClient, task_id: str) -> Dict:
    resp = await client.post(f"{BASE_URL}/reset", json={"task_id": task_id})
    resp.raise_for_status()
    return resp.json()


async def env_step(client: httpx.AsyncClient, action: Dict) -> Dict:
    resp = await client.post(f"{BASE_URL}/step", json={"action": action})
    resp.raise_for_status()
    return resp.json()


async def run_single_task(
    http: httpx.AsyncClient,
    agent: Optional[PolicyEvolverAgent],
    task_id: str,
) -> Dict:
    """Run one task with LLM agent (or rule fallback) and return result."""
    obs = await env_reset(http, task_id)

    if agent is not None:
        if task_id == "task_easy":
            action = agent.handle_easy(obs)
        elif task_id == "task_medium":
            action = agent.handle_medium(obs)
        else:
            action = agent.handle_hard(obs)
        mode = "llm"
    else:
        action = RULE_BASED_ACTIONS[task_id]
        mode = "rule"

    result = await env_step(http, action)
    reward = result.get("reward", 0.0)
    logger.info(f"[{task_id}] mode={mode}  score={reward:.4f}  done={result.get('done')}")
    return {"task_id": task_id, "reward": reward, "mode": mode, "done": result.get("done", False)}


# ─────────────────────────────────────────────
# Direct baseline (no HTTP — used by /baseline endpoint)
# ─────────────────────────────────────────────

async def run_direct_baseline() -> Dict:
    """
    Run baseline directly using environment and grader imports.
    Used by the /baseline endpoint to avoid self-HTTP calls on HF Spaces.
    """
    from server.environment import PolicyEvolverEnvironment
    from server.grader import grade

    env = PolicyEvolverEnvironment()
    use_llm = verify_environment()
    agent = PolicyEvolverAgent() if use_llm else None

    start = time.time()
    results: List[Dict] = []

    for task_id in ["task_easy", "task_medium", "task_hard"]:
        try:
            obs = env.reset(task_id=task_id)
            obs_dict = obs.model_dump()

            if agent is not None:
                if task_id == "task_easy":
                    action = agent.handle_easy(obs_dict)
                elif task_id == "task_medium":
                    action = agent.handle_medium(obs_dict)
                else:
                    action = agent.handle_hard(obs_dict)
                mode = "llm"
            else:
                action = RULE_BASED_ACTIONS[task_id]
                mode = "rule"

            result_obs = env.step(action)
            reward = result_obs.reward
            logger.info(f"[{task_id}] mode={mode}  score={reward:.4f}  done={result_obs.done}")
            results.append({"task_id": task_id, "reward": reward, "mode": mode, "done": result_obs.done})
        except Exception as e:
            logger.error(f"[{task_id}] failed: {e}")
            results.append({"task_id": task_id, "reward": 0.0, "mode": "error", "error": str(e)})

    scores = {r["task_id"]: max(0.0, min(1.0, r["reward"])) for r in results}
    overall = sum(scores.values()) / len(scores) if scores else 0.0

    return {
        "baseline_scores": {
            "task_easy": scores.get("task_easy", 0.0),
            "task_medium": scores.get("task_medium", 0.0),
            "task_hard": scores.get("task_hard", 0.0),
            "overall_avg": round(overall, 4),
        },
        "mode": "llm" if use_llm else "rule_fallback",
        "model": MODEL_NAME if use_llm else "rule-based",
        "runtime_seconds": round(time.time() - start, 2),
        "detail": results,
    }


# ─────────────────────────────────────────────
# Main HTTP-based baseline runner
# ─────────────────────────────────────────────

async def run_llm_baseline() -> Dict:
    """Primary baseline: LLM agent against all 3 tasks via HTTP."""
    use_llm = verify_environment()
    agent = PolicyEvolverAgent() if use_llm else None

    start = time.time()
    results: List[Dict] = []

    async with httpx.AsyncClient(timeout=120.0) as http:
        for task_id in ["task_easy", "task_medium", "task_hard"]:
            if time.time() - start > 1140:
                logger.warning("Approaching 20min time limit — stopping early")
                break
            try:
                r = await run_single_task(http, agent, task_id)
                results.append(r)
            except Exception as e:
                logger.error(f"[{task_id}] failed: {e}")
                results.append({"task_id": task_id, "reward": 0.0, "mode": "error", "error": str(e)})

    scores = {r["task_id"]: max(0.0, min(1.0, r["reward"])) for r in results}
    overall = sum(scores.values()) / len(scores) if scores else 0.0

    summary = {
        "baseline_scores": {
            "task_easy": scores.get("task_easy", 0.0),
            "task_medium": scores.get("task_medium", 0.0),
            "task_hard": scores.get("task_hard", 0.0),
            "overall_avg": round(overall, 4),
        },
        "mode": "llm" if use_llm else "rule_fallback",
        "model": MODEL_NAME if use_llm else "rule-based",
        "runtime_seconds": round(time.time() - start, 2),
        "detail": results,
    }

    # Persist for analysis
    try:
        with open("baseline_results.json", "w") as f:
            json.dump(summary, f, indent=2)
    except Exception:
        pass

    return summary


# Keep rule-based runner available for /baseline endpoint fallback
async def run_rule_based_baseline() -> Dict:
    """Fallback: hardcoded rule-based actions, no LLM required."""
    results: List[Dict] = []
    async with httpx.AsyncClient(timeout=60.0) as http:
        for task_id, action in RULE_BASED_ACTIONS.items():
            try:
                await env_reset(http, task_id)
                result = await env_step(http, action)
                reward = max(0.0, min(1.0, result.get("reward", 0.0)))
                results.append({"task_id": task_id, "reward": reward})
                logger.info(f"[{task_id}] rule score={reward:.4f}")
            except Exception as e:
                logger.error(f"[{task_id}] rule baseline error: {e}")
                results.append({"task_id": task_id, "reward": 0.0})
    scores = {r["task_id"]: r["reward"] for r in results}
    overall = sum(scores.values()) / len(scores) if scores else 0.0
    return {**scores, "overall_avg": round(overall, 4)}


# ─────────────────────────────────────────────
# Rule-based fallback actions (used when OPENAI_API_KEY not set)
# ─────────────────────────────────────────────

RULE_BASED_ACTIONS = {
    "task_easy": {
        "action_type": "propose_clarification",
        "ambiguous_term": "harassment",
        "suggested_definition": (
            "Harassment is defined as any repeated, unwanted communication or behaviour "
            "directed at a specific individual that a reasonable person would find threatening, "
            "intimidating, or distressing. This includes but is not limited to targeted insults, "
            "threats, and sustained negative attention. Single interactions may qualify if "
            "sufficiently severe."
        ),
        "affected_policy_ids": ["pol_002"],
        "justification": (
            "The term 'harassment' is subjective and moderators apply it inconsistently. "
            "Different reviewers may interpret the same post differently without a measurable definition."
        ),
        "think": (
            "Looking at the data, posts 001 and 006 were treated differently despite similar tone. "
            "The key ambiguous term causing inconsistency is 'harassment' in pol_002."
        ),
    },
    "task_medium": {
        "action_type": "propose_new_rule",
        "rule_domain": "AI_use",
        "new_rule": (
            "Employees must disclose when AI tools are used to generate, substantially edit, or "
            "evaluate work products that are submitted under their name, including client proposals, "
            "code submissions, and performance evaluations. AI-assisted content must be reviewed "
            "and validated by the submitting employee before delivery."
        ),
        "scope": [
            "AI-generated client proposals",
            "AI-written code in performance reviews",
            "AI-assisted HR decisions",
            "Automated content in employee-attributed work",
        ],
        "integration_points": ["pol_hr_001", "pol_hr_005"],
        "justification": (
            "Incidents 001, 004, and 007 all involve AI use that current policies do not address. "
            "There is no rule requiring disclosure or validation of AI-generated work, creating "
            "a gap in accountability and intellectual honesty."
        ),
        "think": (
            "The uncovered domain is AI use in professional work. Three of 10 incidents involve this. "
            "The new rule must be mandatory (not advisory) and must specify disclosure + validation."
        ),
    },
    "task_hard": {
        "action_type": "evolve_policy",
        "policy_modifications": [
            {
                "policy_id": "ts_pol_001",
                "change_type": "enhance",
                "new_text": (
                    "New seller accounts with more than 50 transactions in the first week will be "
                    "reviewed only if additional risk signals are present (e.g., chargeback rate > 5%, "
                    "price variance > 30%, or fraud reports). Seasonal categories (gifts, fashion) "
                    "have an elevated threshold of 150 transactions during peak periods."
                ),
                "reason": "Blanket volume threshold causes 42% false positive rate among legitimate high-volume sellers.",
            },
            {
                "policy_id": "ts_pol_002",
                "change_type": "enhance",
                "new_text": (
                    "Return rate thresholds are applied per category: electronics > 10%, fashion > 25%, "
                    "general goods > 15%. Accounts exceeding category thresholds are flagged for review, "
                    "not automatic suspension."
                ),
                "reason": "Return rate varies dramatically by category; a single threshold discriminates against fashion sellers.",
            },
        ],
        "expected_outcomes": {
            "false_positive_rate": 0.20,
            "fraud_detection_rate": 0.35,
            "seller_trust_score": 0.15,
            "review_queue_overload": 0.30,
        },
        "rollback_conditions": [
            "false_positive_rate increases above 0.50 after policy change",
            "fraud_detection_rate drops below 0.25 within 30 days",
            "seller trust score decreases by more than 0.10 in 14-day survey",
        ],
        "justification": (
            "The current framework has a 42% false positive rate because blanket thresholds don't "
            "account for legitimate high-volume or high-return categories. Modifying ts_pol_001 and "
            "ts_pol_002 to be context-aware reduces wrongful suspensions while maintaining fraud "
            "detection via multi-signal scoring. Trade-off: fraud_detection_rate may improve more "
            "slowly since we're relaxing volume triggers, but seller trust and queue overload improve "
            "immediately."
        ),
        "think": (
            "The system_metrics show false_positive_rate=0.42 and fraud_detection_rate=0.31. "
            "The identified issues all point to overly broad thresholds. I should modify the two "
            "most impactful policies and provide category-specific thresholds. "
            "The rollback conditions should be metric-specific with concrete numbers."
        ),
    },
}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PolicyEvolverEnv baseline runner")
    parser.add_argument("--mode", choices=["llm", "rule"], default="llm",
                        help="llm = LLM agent (needs OPENAI_API_KEY); rule = hardcoded fallback")
    parser.add_argument("--output", choices=["text", "json"], default="text")
    args = parser.parse_args()

    if args.mode == "rule":
        summary = asyncio.run(run_rule_based_baseline())
        scores = summary
    else:
        summary = asyncio.run(run_llm_baseline())
        scores = summary.get("baseline_scores", summary)

    if args.output == "json":
        print(json.dumps(summary, indent=2))
    else:
        print("\n" + "=" * 50)
        print("POLICEVOLVERENV BASELINE SCORES")
        print("=" * 50)
        print(f"Easy   (Ambiguity Clarification): {scores.get('task_easy', 0.0):.3f}")
        print(f"Medium (New Rule Proposal):        {scores.get('task_medium', 0.0):.3f}")
        print(f"Hard   (Policy Evolution):         {scores.get('task_hard', 0.0):.3f}")
        print(f"Overall Average:                   {scores.get('overall_avg', 0.0):.3f}")
        print("=" * 50)

        for k, v in scores.items():
            if isinstance(v, float) and not (0.0 <= v <= 1.0):
                raise ValueError(f"Score {k}={v} outside [0.0, 1.0] — submission invalid")
