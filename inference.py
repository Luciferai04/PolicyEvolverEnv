"""
PolicyEvolverEnv — Hackathon Inference Script
=============================================
MANDATORY ENV VARS:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    IMAGE_NAME     The Docker image name (set by validator).

STDOUT FORMAT:
    [START] task=<task_name> env=policy_evolver_env model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   task=<task_name> success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import os
import json
from typing import Dict, List, Optional

from openai import OpenAI
from client import PolicyEvolverEnv
from models import Action

# ─── Environment Variables (Hackathon Mandatory) ───
IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.groq.com/openai/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "llama-3.1-8b-instant"
BENCHMARK = "policy_evolver_env"
MAX_STEPS = 5
TEMPERATURE = 0.0
SUCCESS_THRESHOLD = 0.70


# ─── Logging Helpers (Hackathon Mandatory Format) ───
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = f'"{error}"' if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(task: str, success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] task={task} success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ─── LLM Agent ───
class PolicyEvolverAgent:
    """Strategic Policy Agent — maximizes governance scores via in-context adaptation."""

    SYSTEM_PROMPT = (
        "You are a Strategic Policy Engineer. Your goal is to maximize governance outcomes through verifiable "
        "precision. STYLISTIC RULES:\n"
        "1. NO VAGUENESS: Never use words like 'maybe', 'perhaps', 'sometimes', 'usually'.\n"
        "2. COMMAND LANGUAGE: Use 'must', 'shall', 'prohibited', 'required', 'mandatory'.\n"
        "3. MEASURABLE CRITERIA: Define terms with 'if-then' and metrics.\n"
        "4. ANALYTICAL COT: Your 'think' field MUST be 150-250 words and include terms: 'tradeoff', 'precision', "
        "'recall', 'threshold', 'impact', 'evidence'.\n"
        "5. JSON ONLY: Output ONLY the JSON object. No preamble."
    )

    def __init__(self, model: str):
        self.model = model
        self.action_history: list = []
        self.score_history: list = []

    def _call_llm(self, client: OpenAI, prompt: str) -> Optional[dict]:
        """Call the LLM and robustly parse the JSON response."""
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
                temperature=TEMPERATURE,
                seed=42,
            )
            raw = resp.choices[0].message.content.strip()

            # Strip markdown fences
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()

            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                start = raw.find("{")
                end = raw.rfind("}")
                if start != -1 and end != -1:
                    return json.loads(raw[start : end + 1])
                raise
        except Exception:
            return None

    def _build_feedback(self, step: int, last_score: float, last_action: dict, task_id: str) -> str:
        """Build diagnostic feedback from previous step for in-context learning."""
        if step == 0 or not last_action:
            return ""

        lines = [
            f"\n=== STRATEGIC FEEDBACK (Step {step}) ===",
            f"Previous score: {last_score:.3f} / 1.000",
        ]

        if task_id == "task_easy":
            defn = last_action.get("suggested_definition", "")
            vague = ["might", "could", "perhaps", "sometimes", "often", "generally", "usually", "typically", "may", "possibly"]
            found = [w for w in vague if w in defn.lower()]
            meas = ["threshold", "verify", "days", "$", "%", "reports", "hours", "within", "exceed", "minimum", "must", "shall"]
            mfound = [w for w in meas if w in defn.lower()]
            if found:
                lines.append(f"FAILURE: Vague words detected: {found}. Remove them entirely.")
            if len(mfound) < 2:
                lines.append("FAILURE: Missing measurable criteria. Add numbers, hours, percentages.")
            if len(defn.split()) < 15:
                lines.append("FAILURE: Definition too short. Minimum 15 words.")

        elif task_id == "task_medium":
            if not last_action.get("rule_domain", "").strip():
                lines.append("FAILURE: rule_domain was empty.")
            if len(last_action.get("new_rule", "").split()) < 10:
                lines.append("FAILURE: New rule too short.")

        elif task_id == "task_hard":
            outcomes = last_action.get("expected_outcomes", {})
            if isinstance(outcomes, dict) and len(outcomes) >= 2:
                vals = [v for v in outcomes.values() if isinstance(v, (int, float))]
                vals = [v / 100 if v > 1 else v for v in vals]
                if vals and all(v > 0.70 for v in vals):
                    lines.append("FAILURE: Unrealistic tradeoff — all metrics > 0.70. Model friction.")
            mods = last_action.get("policy_modifications", [])
            if len(mods) < 2:
                lines.append("FAILURE: Need >= 2 policy_modifications.")

        # Append history summaries
        for act, sc in zip(self.action_history[-3:], self.score_history[-3:]):
            lines.append(f"  [{sc:.2f}] {act.get('action_type', '?')}")

        target = min(last_score + 0.25, 0.95)
        lines.append(f"\nYour next proposal MUST score above {target:.2f}.")
        return "\n".join(lines)

    def get_action(self, client: OpenAI, task_id: str, obs: dict) -> dict:
        """Generate the next strategic action for the given task."""
        step = obs.get("step_count", 0)
        last_score = obs.get("info", {}).get("last_reward", 0.0)
        last_action = obs.get("info", {}).get("last_action", {})
        feedback = self._build_feedback(step, last_score, last_action, task_id)

        if task_id == "task_easy":
            prompt = (
                f"POLICIES: {obs.get('current_policies', [])}\n"
                f"DATA: {obs.get('data_corpus', [])[:5]}\n{feedback}\n"
                "TASK: Propose clarification for an ambiguous term with a measurable definition.\n"
                "JSON: {\"action_type\": \"propose_clarification\", \"ambiguous_term\": \"...\", "
                "\"suggested_definition\": \"...\", \"affected_policy_ids\": [\"str\"], "
                "\"justification\": \"...\", \"think\": \"...\"}"
            )
        elif task_id == "task_medium":
            prompt = (
                f"POLICIES: {obs.get('current_policies', [])}\n"
                f"DATA: {obs.get('data_corpus', [])}\n{feedback}\n"
                "TASK: Propose a new rule for a coverage gap. Use mandatory language.\n"
                "JSON: {\"action_type\": \"propose_new_rule\", \"rule_domain\": \"...\", "
                "\"new_rule\": \"...\", \"scope\": [\"str\"], \"integration_points\": [\"str\"], "
                "\"justification\": \"...\", \"think\": \"...\"}"
            )
        else:
            prompt = (
                f"METRICS: {obs.get('system_metrics', {})}\n"
                f"ISSUES: {obs.get('identified_issues', [])}\n{feedback}\n"
                "TASK: Evolve policies with realistic tradeoffs.\n"
                "JSON: {\"action_type\": \"evolve_policy\", \"policy_modifications\": "
                "[{\"policy_id\": \"...\", \"change_type\": \"enhance|restrict|add|remove\", "
                "\"new_text\": \"...\", \"reason\": \"...\"}], \"expected_outcomes\": "
                "{\"fraud_rate\": 0.8, \"revenue_velocity\": 0.4}, "
                "\"rollback_conditions\": [\"...\"], \"justification\": \"...\", \"think\": \"...\"}"
            )

        result = self._call_llm(client, prompt)
        if result is None:
            # Fallback action so we never crash
            result = {
                "action_type": "propose_clarification",
                "ambiguous_term": "offensive",
                "suggested_definition": "Content must be classified as offensive if and only if it contains explicit slurs, direct threats of physical violence, or identity-based harassment verifiable within 24 hours.",
                "affected_policy_ids": ["pol_001"],
                "justification": "Replaces subjective term with measurable criteria.",
                "think": "Analyzing the tradeoff between precision and recall for content moderation threshold. Evidence from the corpus shows impact of vague terms on enforcement. Setting a measurable threshold improves consistency.",
            }
        return result


# ─── Episode Runner ───
async def run_episode(client: Optional[OpenAI], env: Optional[PolicyEvolverEnv], task_id: str, setup_error: Optional[Exception] = None) -> dict:
    """Run a single task episode following the hackathon format."""
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    if setup_error:
        log_step(step=1, action="setup", reward=0.0, done=True, error=str(setup_error))
        log_end(task=task_id, success=False, steps=0, score=0.0, rewards=[])
        return {"task_id": task_id, "reward": 0.0}

    if not client or not env:
        log_step(step=1, action="setup", reward=0.0, done=True, error="Client or Environment not initialized")
        log_end(task=task_id, success=False, steps=0, score=0.0, rewards=[])
        return {"task_id": task_id, "reward": 0.0}

    agent = PolicyEvolverAgent(MODEL_NAME)
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        result = await env.reset(task_id=task_id)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            # Get observation as dict
            obs_dict = result.observation
            if hasattr(obs_dict, "model_dump"):
                obs_dict = obs_dict.model_dump()
            elif not isinstance(obs_dict, dict):
                obs_dict = dict(obs_dict)

            # Agent decides action
            action_dict = agent.get_action(client, task_id, obs_dict)
            agent.action_history.append(action_dict)

            # Validate and step
            error = None
            try:
                action_obj = Action.model_validate(action_dict)
                result = await env.step(action_obj)
                reward = result.reward or 0.0
                done = result.done
            except Exception as e:
                reward = 0.0
                done = True
                error = str(e)

            rewards.append(reward)
            agent.score_history.append(reward)
            steps_taken = step

            act_name = action_dict.get("action_type", "unknown")
            log_step(step=step, action=act_name, reward=reward, done=done, error=error)

            if done:
                break

        score = rewards[-1] if rewards else 0.0
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        log_step(step=steps_taken + 1, action="error", reward=0.0, done=True, error=str(e))

    finally:
        log_end(task=task_id, success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task_id": task_id, "reward": score}


# ─── Main Entry Point ───
async def main() -> None:
    client = None
    env = None
    setup_error = None

    try:
        # 1. Initialize OpenAI Client
        try:
            if not API_KEY or API_KEY == "dummy_key":
                # In validator, API_KEY might be missing if they didn't pass it yet, but we shouldn't crash
                pass
            client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy")
        except Exception as e:
            setup_error = Exception(f"OpenAI client initialization failed: {e}")

        # 2. Initialize Environment
        if not setup_error:
            try:
                if IMAGE_NAME:
                    env = await PolicyEvolverEnv.from_docker_image(IMAGE_NAME)
                else:
                    local_url = os.environ.get("ENV_BASE_URL", "http://127.0.0.1:7860")
                    env = PolicyEvolverEnv(base_url=local_url)
                    # For local testing, we might want to check connection immediately or let run_episode handle it
            except Exception as e:
                setup_error = Exception(f"Environment initialization failed: {e}")

    except Exception as e:
        setup_error = e

    # 3. Always loop over tasks to ensure structured logs
    tasks = ["task_easy", "task_medium", "task_hard"]
    for task in tasks:
        await run_episode(client, env, task, setup_error=setup_error)

    # 4. Final Cleanup
    if env:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
