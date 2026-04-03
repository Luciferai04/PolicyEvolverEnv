import os
import json
import time
from typing import Dict, List, Optional
from openai import OpenAI

# ─────────────────────────────────────────────
# Mandatory Fix B: Standardized Environment Variables
# ─────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")

# Unified client construction as per Fix B instructions
llm_client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL,
)

class PolicyEvolverAgent:
    """Standalone agent for hackathon inference."""
    def __init__(self, model: str):
        self.model = model

    def _call(self, prompt: str) -> Optional[Dict]:
        try:
            resp = llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a senior policy analyst. Respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            raw = resp.choices[0].message.content.strip()
            # Clean possible markdown
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()
            return json.loads(raw)
        except Exception as e:
            # Fallback to a structured error action to prevent breakdown
            return None

    def _get_history(self, obs: Dict) -> str:
        info = obs.get("info", {})
        if obs.get("step_count", 0) == 0: return ""
        return f"\nPREVIOUS STEP: Score={info.get('last_reward', 0):.2f}. Actions: {info.get('action_history', [])}\n"

    def act(self, task_id: str, obs: Dict) -> Dict:
        history = self._get_history(obs)
        if task_id == "task_easy":
            prompt = f"Policies: {obs['current_policies']}\nData: {obs['data_corpus'][:5]}\n{history}\nTask: Propose clarification for an ambiguous term. Respond with JSON: {{'action_type': 'propose_clarification', 'ambiguous_term': '...', 'suggested_definition': '...', 'affected_policy_ids': ['str'], 'justification': '...'}}"
        elif task_id == "task_medium":
            prompt = f"Policies: {obs['current_policies']}\nData: {obs['data_corpus']}\n{history}\nTask: Propose a new rule for a gap. Respond with JSON: {{'action_type': 'propose_new_rule', 'rule_domain': '...', 'new_rule': '...', 'scope': ['str'], 'integration_points': ['str'], 'justification': '...'}}"
        else:
            prompt = f"Metrics: {obs['system_metrics']}\nIssues: {obs['identified_issues']}\n{history}\nTask: Evolve policies for better performance. Respond with exactly this JSON structure: {{'action_type': 'evolve_policy', 'policy_modifications': [{{'policy_id': 'id_here', 'change_type': 'enhance|restrict|add|remove', 'new_text': '...', 'reason': '...'}}], 'expected_outcomes': {{'false_positive_rate': -0.1}}, 'rollback_conditions': ['condition 1 as string'], 'justification': '...'}}"
        
        action = self._call(prompt) or {"action_type": "propose_clarification", "ambiguous_term": "NONE", "suggested_definition": "NONE", "affected_policy_ids": [], "justification": "ERROR"}
        return action

def run_episode(task_id: str):
    # Fix: Import environment within loop to ensure clean isolation
    from server.environment import PolicyEvolverEnvironment
    from models import Action
    
    env = PolicyEvolverEnvironment()
    agent = PolicyEvolverAgent(MODEL_NAME)
    
    # [START] line - Hackathon Mandatory Format
    print(f"[START] task={task_id} env=PolicyEvolverEnv model={MODEL_NAME}", flush=True)
    
    obs = env.reset(task_id=task_id)
    step_num = 0
    rewards = []
    success = False
    
    # Strategic refinement for 3 steps (Fix C: Limit steps for 20min run)
    for _ in range(3):
        step_num += 1
        action_dict = agent.act(task_id, obs.model_dump())
        
        obs = env.step(Action.model_validate(action_dict))
        
        reward = obs.reward
        done = obs.done
        rewards.append(reward)
        
        # [STEP] line: Hackathon Mandatory Format
        action_name = action_dict.get("action_type", "unknown")
        print(f"[STEP] step={step_num} action={action_name} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
        
        if done:
            success = reward >= 0.70
            break
            
    # [END] line - Hackathon Mandatory Format
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    score = rewards[-1] if rewards else 0.0
    print(f"[END] success={str(success).lower()} steps={step_num} score={score:.3f} rewards={rewards_str}", flush=True)
    return {"task_id": task_id, "reward": rewards[-1], "steps": step_num}

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", choices=["text", "json"], default="text")
    parser.add_argument("task", nargs="?", default=None)
    args = parser.parse_args()

    results = []
    tasks = [args.task] if args.task else ["task_easy", "task_medium", "task_hard"]
    
    start_time = time.time()
    for t in tasks:
        try:
            res = run_episode(t)
            results.append(res)
        except Exception as e:
            print(f"[END] success=false steps=0 rewards=0.00 error={str(e)}")
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
