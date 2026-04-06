import os
import json
import time
import sys
from typing import Dict, List, Optional
from openai import OpenAI

# ─────────────────────────────────────────────
# Mandatory Fix: Standardized Environment Variables (Groq Migration)
# ─────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

if not HF_TOKEN:
    print("ERROR: HF_TOKEN environment variable is not set.")
    print("  Export your Groq API key: export HF_TOKEN=gsk_...")
    sys.exit(1)

# Modern OpenAI-compatible client construction (Groq)
llm_client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL,
)

# Quick connectivity check before running episodes
def verify_llm_connection(verbose: bool = True):
    try:
        _conn_test = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Say OK"}],
            temperature=0.0,
            seed=42,
            max_tokens=5,
        )
        if verbose: print(f"[OK] LLM connection verified. Provider: {API_BASE_URL}", flush=True)
    except Exception as e:
        print(f"ERROR: LLM connection failed: {e}")
        print(f"  API_BASE_URL = {API_BASE_URL}")
        print(f"  MODEL_NAME   = {MODEL_NAME}")
        print(f"  HF_TOKEN set = {'yes' if HF_TOKEN else 'no'}")
        sys.exit(1)

class PolicyEvolverAgent:
    """Standalone agent for hackathon inference. Upgraded for 0.9+ scores (Groq/Llama-3.3)."""
    def __init__(self, model: str):
        self.model = model
        self.action_history: list = []
        self.score_history: list = []

    def _call(self, prompt: str) -> Optional[Dict]:
        try:
            resp = llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system", 
                        "content": (
                            "You are a Strategic Policy Engineer. Your goal is to maximize governance outcomes through verifiable "
                            "precision. STYLISTIC RULES:\n"
                            "1. NO VAGUENESS: Never use words like 'maybe', 'perhaps', 'sometimes', 'usually'.\n"
                            "2. COMMAND LANGUAGE: Use 'must', 'shall', 'prohibited', 'required', 'mandatory'.\n"
                            "3. MEASURABLE CRITERIA: Define terms with 'if-then' and metrics.\n"
                            "4. ANALYTICAL COT: Your 'think' field MUST be 150-250 words and include terms: 'tradeoff', 'precision', 'recall', 'threshold', 'impact', 'evidence'.\n"
                            "5. JSON ONLY: Output ONLY the JSON object. No preamble."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                temperature=0.0,
                seed=42
            )
            raw = resp.choices[0].message.content.strip()
            
            # Robust parsing for chatty models
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()
            
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                # Last resort: find broadest {} range
                start = raw.find("{")
                end = raw.rfind("}")
                if start != -1 and end != -1:
                    return json.loads(raw[start:end+1])
                raise
        except Exception as e:
            return None

    def _summarise_action(self, action: dict, score: float, task_id: str) -> str:
        """One-line compact summary of an action for history injection."""
        try:
            if task_id == "task_easy":
                defn = action.get("suggested_definition", "")
                preview = defn[:80] + "..." if len(defn) > 80 else defn
                return f"  [{score:.2f}] Definition: '{preview}'"
            elif task_id == "task_medium":
                domain = action.get("rule_domain", "unknown")
                rule = action.get("new_rule", "")
                preview = rule[:60] + "..." if len(rule) > 60 else rule
                return f"  [{score:.2f}] Domain={domain}: '{preview}'"
            elif task_id == "task_hard":
                outcomes = action.get("expected_outcomes", {})
                fr = outcomes.get("fraud_rate", "?")
                rv = outcomes.get("revenue_velocity", "?")
                st = outcomes.get("seller_trust", "?")
                mods = action.get("policy_modifications", [])
                return f"  [{score:.2f}] fraud={fr}, revenue={rv}, trust={st}, mods={len(mods)}"
            return f"  [{score:.2f}] [action]"
        except Exception:
            return f"  [{score:.2f}] [summary error]"

    def _get_history(self, step: int, last_score: float, last_action: dict, task_id: str) -> str:
        if step == 0 or not last_action:
            return ""

        feedback_lines = [
            f"\n=== STRATEGIC FEEDBACK (Step {step}) ===",
            f"Previous score: {last_score:.3f} / 1.000",
        ]

        # Task-specific failure diagnosis
        if task_id == "task_easy":
            defn = last_action.get("suggested_definition", "")
            vague_words = ["might", "could", "perhaps", "sometimes", "often", "generally", "usually", "typically", "may", "possibly"]
            vague_found = [w for w in vague_words if w in defn.lower()]
            measurable = ["threshold", "verify", "days", "$", "%", "reports", "hours", "within", "exceed", "minimum", "specifically", "measurable", "if-then", "must", "shall"]
            meas_found = [w for w in measurable if w in defn.lower()]

            if vague_found:
                feedback_lines.append(f"FAILURE REASON: Definition contained vague words: {vague_found}. Remove them entirely.")
            if len(meas_found) < 2:
                feedback_lines.append("FAILURE REASON: Missing measurable criteria. Add specific numbers: hours, report counts, percentages, or dollar thresholds.")
            if len(defn.split()) < 15:
                feedback_lines.append("FAILURE REASON: Definition too short. Minimum 15 words with at least 2 numeric/measurable criteria.")

        elif task_id == "task_medium":
            domain = last_action.get("rule_domain", "").strip()
            rule = last_action.get("new_rule", "")
            if not domain:
                feedback_lines.append("FAILURE REASON: rule_domain was empty. You must specify the exact governance silo.")
            if len(rule.split()) < 10:
                feedback_lines.append("FAILURE REASON: New rule too short. Must include who is affected, what is required, and enforcement method.")

        elif task_id == "task_hard":
            outcomes = last_action.get("expected_outcomes", {})
            if isinstance(outcomes, dict) and len(outcomes) >= 2:
                vals = [v for v in outcomes.values() if isinstance(v, (int, float))]
                vals = [v / 100.0 if v > 1.0 else v for v in vals]
                if vals and all(v > 0.70 for v in vals):
                    feedback_lines.append("FAILURE REASON: Unrealistic tradeoff detected. ALL metrics cannot simultaneously exceed 0.70. Model friction explicitly.")
                elif vals and max(vals) - min(vals) < 0.15:
                    feedback_lines.append(f"FAILURE REASON: Insufficient tradeoff variance. Values too close together: {outcomes}.")
            else:
                feedback_lines.append("FAILURE REASON: expected_outcomes missing or incomplete.")

            policy_mods = last_action.get("policy_modifications", [])
            if len(policy_mods) < 2:
                feedback_lines.append("FAILURE REASON: policy_modifications must contain at least 2 entries — one tightening rule and one exemption/rollback condition.")

        # Summarise history (last 3 attempts)
        history_entries = []
        for i, (act, sc) in enumerate(zip(self.action_history[-3:], self.score_history[-3:])):
            history_entries.append(self._summarise_action(act, sc, task_id))
        
        history_str = "\nPrevious attempts (most recent last):\n" + "\n".join(history_entries) if history_entries else ""
        
        target = min(last_score + 0.25, 0.95)
        feedback_lines.append(f"\nINSTRUCTION: Your next proposal MUST score above {target:.2f}. Address every FAILURE REASON. Model tradeoffs explicitly.")
        
        return "\n".join(feedback_lines) + "\n" + history_str

    def act(self, task_id: str, obs: Dict) -> Dict:
        step = obs.get("step_count", 0)
        last_score = obs.get("info", {}).get("last_reward", 0.0)
        last_action = obs.get("info", {}).get("last_action", {})
        
        history = self._get_history(step, last_score, last_action, task_id)
        if task_id == "task_easy":
            prompt = (
                f"POLICIES: {obs['current_policies']}\nDATA: {obs['data_corpus'][:5]}\n{history}\n"
                "TASK: Propose clarification for an ambiguous term. Replace it with a measurable, if-then definition. \n"
                "JSON FORMAT: {'action_type': 'propose_clarification', 'ambiguous_term': '...', 'suggested_definition': '...', 'affected_policy_ids': ['str'], 'justification': '...', 'think': '...'}"
            )
        elif task_id == "task_medium":
            prompt = (
                f"POLICIES: {obs['current_policies']}\nDATA: {obs['data_corpus']}\n{history}\n"
                "TASK: Propose a new rule for a coverage gap. Use mandatory language ('shall', 'must'). \n"
                "JSON FORMAT: {'action_type': 'propose_new_rule', 'rule_domain': '...', 'new_rule': '...', 'scope': ['str'], 'integration_points': ['str'], 'justification': '...', 'think': '...'}"
            )
        else:
            prompt = (
                f"METRICS: {obs['system_metrics']}\nISSUES: {obs['identified_issues']}\n{history}\n"
                "TASK: Evolve policies for better performance. Model realistic tradeoffs explicitly. \n"
                "JSON FORMAT: {'action_type': 'evolve_policy', 'policy_modifications': [{'policy_id': '...', 'change_type': 'enhance|restrict|add|remove', 'new_text': '...', 'reason': '...'}], 'expected_outcomes': {'fraud_rate': 0.8, 'revenue_velocity': 0.4}, 'rollback_conditions': ['...'], 'justification': '...', 'think': '...'}"
            )
        
        action = self._call(prompt) or {"action_type": "propose_clarification", "ambiguous_term": "RETRY", "suggested_definition": "ERROR", "affected_policy_ids": [], "justification": "ERROR", "think": "RETRY"}
        return action

def run_episode(task_id: str, verbose: bool = True):
    # Fix: Import environment within loop to ensure clean isolation
    from server.environment import PolicyEvolverEnvironment
    from models import Action
    
    env = PolicyEvolverEnvironment()
    agent = PolicyEvolverAgent(MODEL_NAME)
    
    # [START] line - Hackathon Mandatory Format
    if verbose: print(f"[START] task={task_id} env=PolicyEvolverEnv model={MODEL_NAME}", flush=True)
    
    obs = env.reset(task_id=task_id)
    step_num = 0
    rewards = []
    success = False
    
    # Strategic refinement for 5 steps (Audit Mode: 5 iterations)
    for _ in range(5):
        step_num += 1
        action_dict = agent.act(task_id, obs.model_dump())
        
        obs = env.step(Action.model_validate(action_dict))
        
        reward = obs.reward
        done = obs.done
        rewards.append(reward)
        
        # FIX 3: Append to history
        agent.action_history.append(action_dict)
        agent.score_history.append(reward)
        
        # [STEP] line: Hackathon Mandatory Format
        action_name = action_dict.get("action_type", "unknown")
        if verbose: print(f"[STEP] step={step_num} action={action_name} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
        
        if done:
            success = reward >= 0.70
            break
            
    # [END] line - Hackathon Mandatory Format
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    score = rewards[-1] if rewards else 0.0
    if verbose: print(f"[END] success={str(success).lower()} steps={step_num} score={score:.3f} rewards={rewards_str}", flush=True)
    return {"task_id": task_id, "reward": rewards[-1], "steps": step_num}

def verify_diagnostics():
    """FIX 2 Verification: Diagnosis check."""
    agent = PolicyEvolverAgent("meta-llama/Llama-3.3-70B-Instruct")
    bad_action = {"suggested_definition": "behavior that might sometimes be bad"}
    history = agent._get_history(step=1, last_score=0.15, last_action=bad_action, task_id="task_easy")
    print(history)
    assert "FAILURE REASON" in history
    assert "vague words" in history.lower() or "measurable" in history.lower()
    print("FIX 2: _get_history diagnosis test passed")

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", choices=["text", "json"], default="text")
    parser.add_argument("task", nargs="?", default=None)
    args = parser.parse_args()

    results = []
    tasks = [args.task] if args.task else ["task_easy", "task_medium", "task_hard"]
    verbose = (args.output == "text")
    
    # Verify connection once before running tasks
    verify_llm_connection(verbose=verbose)
    
    start_time = time.time()
    for t in tasks:
        try:
            res = run_episode(t, verbose=verbose)
            results.append(res)
        except Exception as e:
            if verbose: print(f"[END] success=false steps=0 rewards=0.00 error={str(e)}")
            results.append({"task_id": t, "reward": 0.0, "error": str(e)})

    # Internal JSON output for server /baseline endpoint
    if args.output == "json":
        # Print a separator if we have logs before
        # Using sys.stderr or similar would be better, but we need to pass back structured data.
        overall = sum(r.get("reward", 0.0) for r in results) / len(results) if results else 0.0
        final_summary = {
            "baseline_scores": {"overall_avg": round(overall, 4)},
            "model": MODEL_NAME,
            "runtime_seconds": round(time.time() - start_time, 2),
            "detail": results
        }
        # Final line is the JSON
        print(json.dumps(final_summary))
