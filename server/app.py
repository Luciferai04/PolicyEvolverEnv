# server/app.py
# HF Force Rebuild: 2026-03-30T17:18:00Z
from __future__ import annotations
import os
import json
import traceback
import uvicorn
import pandas as pd
import gradio as gr
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from openenv.core.env_server import create_fastapi_app
from models import (
    ProposeClarificationAction, ProposeNewRuleAction, EvolveProcessAction,
    Observation, Action, PolicyActionType, TaskInfo
)
from server.environment import PolicyEvolverEnvironment
from server.grader import grade
from server.tasks import TASK_REGISTRY

# Initialize Environment and FastAPI app
env = PolicyEvolverEnvironment()
app = create_fastapi_app(
    env=PolicyEvolverEnvironment,
    action_cls=Action,
    observation_cls=Observation,
)

# Remove default routes to avoid collision with custom overrides below
app.router.routes = [r for r in app.router.routes if r.path not in ["/health", "/state", "/tasks", "/grader", "/baseline"]]

# Custom Exception Handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": "Invalid Action", "errors": exc.errors()})

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Error", "message": str(exc), "traceback": traceback.format_exc()},
    )

@app.get("/")
async def root():
    """Root endpoint for automated pings to return 200 OK."""
    return {"message": "PolicyEvolverEnv is running", "status": "ok"}

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/state")
def get_state():
    """Return the current environment state."""
    return {
        "episode_id": env.state.episode_id,
        "step_count": env.state.step_count,
        "max_steps": env.state.max_steps,
        "current_score": env.state.current_score
    }


@app.get("/tasks")
def list_tasks() -> list[TaskInfo]:
    """Return all tasks with their action schema."""
    return [
        TaskInfo(
            task_id=tid,
            difficulty=task["difficulty"],
            description=task["description"],
            action_schema=Action.model_json_schema(),
        )
        for tid, task in TASK_REGISTRY.items()
    ]


@app.post("/grader")
def get_grader_score(task_id: str, action: dict):
    """
    Grade a submission directly.
    """
    if task_id not in TASK_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id}")
    
    score = grade(action, task_id)
    return {
        "task_id": task_id,
        "score": score,
        "passed": 1 if score > 0.5 else 0, # Hackathon-appropriate proxy
        "total": 1
    }


@app.get("/baseline")
def run_baseline_route():
    """
    Run the baseline agent on all tasks and return scores.
    """
    import subprocess, sys, os
    try:
        # Inherit required env vars
        env_vars = os.environ.copy()
        # Fix A: Call root-level inference.py
        result = subprocess.run(
            [sys.executable, "inference.py", "--output", "json"],
            capture_output=True, 
            text=True, 
            timeout=180,
            env=env_vars
        )
        raw = json.loads(result.stdout)
        # Map to required structure: {"baseline_results": [...], "average_score": float, "model": ...}
        return {
            "baseline_results": raw.get("detail", []),
            "average_score": raw.get("baseline_scores", {}).get("overall_avg", 0.0),
            "model": raw.get("model", "llama-3.3-70b-versatile")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ───────────────────────────────────────────────────────────────────────────
# Custom Professional "Judge Ready" Gradio Dashboard
# ───────────────────────────────────────────────────────────────────────────

def build_custom_ui():
    env = PolicyEvolverEnvironment()

    def format_obs(obs):
        """Helper to extract tabular data and Markdown for the Judge Console."""
        if not obs: return pd.DataFrame(), "### No Data Framework Available", 0.0, 5, "N/A"
        
        # 1. Data Corpus Table (Dynamic Handling)
        corpus_data = []
        for item in obs.get("data_corpus", []):
            content = item.get("content") or item.get("text") or item.get("type", "N/A") 
            if "flags" in item:
                content += f" | Tags: {', '.join(item['flags'])}"
            if "desc" in item:
                content += f" | Info: {item['desc']}"
            
            corpus_data.append({
                "ID": item.get("id"),
                "Content": content[:120] + ("..." if len(content) > 120 else ""),
                "System Action": item.get("system_action") or item.get("action_taken") or item.get("outcome", "pending")
            })
        df_corpus = pd.DataFrame(corpus_data) if corpus_data else pd.DataFrame(columns=["ID", "Content", "System Action"])

        # 2. Policy List (Markdown)
        policy_md = "### 📜 Active Governance Framework\n"
        for p in obs.get("current_policies", []):
            policy_md += f"- **{p.get('id')}**: {p.get('text')}\n"

        # 3. Simple Stats & Reward History
        history = obs.get("info", {}).get("rewards_history", [])
        df_reward = pd.DataFrame({
            "Step": [i + 1 for i in range(len(history))],
            "Reward": history
        })
        if df_reward.empty:
            df_reward = pd.DataFrame({"Step": [0], "Reward": [0.0]})

        best_score = obs.get("info", {}).get("best_score", 0.0)
        steps_left = obs.get("info", {}).get("steps_remaining", 5)
        episode_id = obs.get("episode_id", "N/A")[:8]
        
        shown = obs.get("corpus_shown", len(corpus_data))
        total = obs.get("corpus_size", len(corpus_data))
        corpus_stat = f"### 📊 Corpus: **{shown}** of **{total}** incidents displayed"
        
        return df_corpus, policy_md, best_score, steps_left, episode_id, corpus_stat, df_reward

    def handle_reset(task_id):
        obs = env.reset(task_id=task_id).model_dump()
        df, pol, score, steps, ep, stat, df_hist = format_obs(obs)
        reward_msg = "### 🏁 Scenario Initialized\nReview the Data Corpus and Active Framework to identify gaps."
        return df, pol, score, steps, ep, stat, df_hist, reward_msg, json.dumps(obs, indent=2)

    def handle_step(task_id, action_type, easy_term, easy_def, easy_just, easy_think,
                    med_domain, med_rule, med_scope, med_just, med_think,
                    hard_mods, hard_outcomes, hard_just, hard_think):
        try:
            payload = {"action_type": action_type}
            if action_type == "propose_clarification":
                payload.update({"ambiguous_term": easy_term or "", "suggested_definition": easy_def or "", "justification": easy_just or "", "think": easy_think or "", "affected_policy_ids": ["pol_001"]})
            elif action_type == "propose_new_rule":
                payload.update({"rule_domain": med_domain or "", "new_rule": med_rule or "", "scope": [s.strip() for s in (med_scope or "").split(",") if s.strip()], "justification": med_just or "", "think": med_think or ""})
            elif action_type == "evolve_policy":
                payload.update({"policy_modifications": json.loads(hard_mods) if hard_mods else [], "expected_outcomes": json.loads(hard_outcomes) if hard_outcomes else {}, "justification": hard_just or "", "think": hard_think or ""})

            validated_action = Action.model_validate(payload)
            obs_obj = env.step(validated_action)
            obs = obs_obj.model_dump()
            df, pol, score, steps, ep, stat, df_hist = format_obs(obs)
            
            reward = obs.get("reward", 0.0)
            color = "green" if reward > 0 else "orange" if reward == 0 else "red"
            reward_msg = f"### <span style='color:{color}'>Latest Strategic Reward: {reward}</span>\nCurrent Project Score: {score}"
            
            return df, pol, score, steps, ep, stat, df_hist, reward_msg, json.dumps(obs, indent=2)
        except Exception as e:
            return pd.DataFrame(), f"### Execution Error\n{str(e)}", 0, 0, "ERROR", "### ERROR", pd.DataFrame(), f"Traceback:\n{traceback.format_exc()}", "{}"

    with gr.Blocks(
        title="PolicyEvolver Judge Console",
    ) as demo:
        gr.HTML("<h1 style='text-align: center; color: #2D5A27;'>PolicyEvolver: Judge's Strategic Console</h1>")
        gr.Markdown("Welcome, Judge Agent. Use this console to identify data-to-policy gaps and propose measurable governance refinements.")

        with gr.Row():
            # LEFT: Leaderboard & Meta-Data
            with gr.Column(scale=1, variant="panel"):
                gr.Markdown("### 📈 Scenario Metrics")
                best_score_disp = gr.Number(label="Environment Best Score", value=0.0, interactive=False)
                steps_left_disp = gr.Number(label="Remaining Execution Steps", value=5, interactive=False)
                episode_disp = gr.Textbox(label="Active Episode ID", value="N/A", interactive=False)
                
                gr.Markdown("### 📈 Reward Evolution")
                reward_plot = gr.LinePlot(
                    label="Strategic Reward Trend",
                    x="Step",
                    y="Reward",
                    tooltip=["Step", "Reward"],
                )
                
                reward_outcome_disp = gr.Markdown("### Awaiting Scenario...")
                
                gr.Markdown("---")
                task_id = gr.Dropdown(choices=list(TASK_REGISTRY.keys()), value="task_easy", label="Deployment Scenario")
                reset_btn = gr.Button("Initialize Scenario", variant="secondary")

            # RIGHT: Observations & Data Corpus
            with gr.Column(scale=3):
                corpus_count_disp = gr.Markdown("### 📊 Corpus: 0 of 0 incidents displayed")
                with gr.Tabs():
                    with gr.Tab("📋 Data Corpus (Tabular View)"):
                        corpus_table = gr.DataFrame(label="Sampled Posts and System Actions", interactive=False)
                    with gr.Tab("📜 Active Framework"):
                        policy_display = gr.Markdown("Initialize to see current active framework.")
                    with gr.Tab("🔍 Diagnostic JSON"):
                        raw_json_box = gr.Code(label="Environment Raw Response", language="json", interactive=False)

        gr.Markdown("---")
        
        # BOTTOM: Action Console
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Propose Strategic Refinement")
                action_mode = gr.Radio(
                    choices=[("1. Clarification (Easy)", "propose_clarification"), ("2. New Rule (Medium)", "propose_new_rule"), ("3. Evolution (Hard)", "evolve_policy")],
                    value="propose_clarification",
                    label="Current Execution Mode"
                )
                
                with gr.Tabs() as action_tabs:
                    with gr.Tab("Easy: Definition Refining"):
                        gr.Markdown("*Fix subjectivity by precisely defining ambiguous terms from the corpus.*")
                        with gr.Row():
                            load_easy_btn = gr.Button("Load Expert Suggestion", variant="secondary", size="sm")
                        easy_term = gr.Textbox(label="Target Ambiguous Term", placeholder="e.g. offensive")
                        easy_def = gr.TextArea(label="Proposed Specific Definition", placeholder="Be precise and measurable...")
                        easy_just = gr.TextArea(label="Justification", placeholder="How does this fix specific items in the data?")
                        easy_think = gr.Textbox(label="Agent Reasoning (CoT)", placeholder="Internal logic...")
                        
                        def load_easy():
                            return (
                                "offensive",
                                "Content is defined as offensive if it includes explicit slurs, direct insults targeting protected identity characteristics, or specific threats of physical violence. It refers to content that is measurable through community guidelines and will be removed.",
                                "The current policy leads to inconsistent moderation because the term is subjective. Moderators interpret it differently which causes significant disputes.",
                                "I am narrowing the definition to measurable slurs and insults to remove subjectivity and ensure consistency across human moderators."
                            )
                        load_easy_btn.click(load_easy, outputs=[easy_term, easy_def, easy_just, easy_think])

                    with gr.Tab("Medium: Gap Detection"):
                        gr.Markdown("*Propose entire new rules for detected coverage gaps.*")
                        with gr.Row():
                            load_med_btn = gr.Button("Load Expert Suggestion", variant="secondary", size="sm")
                        med_domain = gr.Textbox(label="Risk Domain", placeholder="e.g. AI-generated hate speech")
                        med_rule = gr.TextArea(label="Draft New Rule Text", placeholder="Draft the complete policy text...")
                        med_scope = gr.Textbox(label="Applicable Context Tags", placeholder="images, chat, user_meta...")
                        med_just = gr.TextArea(label="Evidence of Coverage Gap", placeholder="Evidence for why this rule is needed...")
                        med_think = gr.Textbox(label="Agent Reasoning (CoT)", placeholder="Explain your logic...")

                        def load_med(task_id):
                            if task_id == "task_hard":
                                return (
                                    "seller_legitimacy",
                                    "Sellers with fewer than 30 days of history and more than 20 sales per day must complete enhanced identity verification before withdrawals are processed.",
                                    "marketplace, fraud, seller_onboarding, payments",
                                    "Cases h_leg_001 and h_leg_005 show that rapid sales velocity combined with zero return history is a known fraud pattern not covered by current policies.",
                                    "The corpus shows multiple high-velocity new seller patterns. The gap is the absence of velocity-based verification triggers in the onboarding policy."
                                )
                            return (
                                "AI_use",
                                "Employees must explicitly disclose any use of generative AI tools when drafting client proposals or proprietary code. This requirement is mandatory and will be monitored through manual reviews.",
                                "chat, code, email, documents",
                                "Current policies like pol_hr_001 handle general confidentiality but do not account for data privacy risks specifically associated with external AI training sets.",
                                "I am bridging the gap between general confidentiality and AI usage. By introducing mandatory disclosure, we mitigate the risk of proprietary data leakages."
                            )
                        load_med_btn.click(load_med, inputs=[task_id], outputs=[med_domain, med_rule, med_scope, med_just, med_think])

                    with gr.Tab("Hard: Full System Evolution"):
                        gr.Markdown("*Manually modify the underlying framework logic.*")
                        with gr.Row():
                            load_hard_btn = gr.Button("Load Expert Suggestion", variant="secondary", size="sm")
                        hard_mods = gr.TextArea(label="Policy Mods (JSON Array)", value="[]")
                        hard_outcomes = gr.TextArea(label="Projected Impact (JSON Dict)", value="{}")
                        hard_just = gr.TextArea(label="Strategic Rationale", placeholder="Comprehensive reasoning...")
                        hard_think = gr.Textbox(label="Agent Reasoning (CoT)", placeholder="Explain your logic...")

                        def load_hard():
                            return (
                                '[{"policy_id": "pol_rev_001", "change_type": "enhance", "new_text": "Apply manual review thresholds for high-volume cross-border merchants.", "reason": "Targeting category-specific fraud spikes."}]',
                                '{"fraud_rate": 0.15, "revenue_velocity": 0.20, "seller_trust": 0.10}',
                                "We are balancing precision and recall by isolating high-volume risk categories while rewarding legitimate legacy sellers. This address the trade-off between strict fraud detection and overall revenue growth.",
                                "I am optimizing the framework to reduce false positives for trusted sellers while tightening the manual review net for high-risk categories."
                            )
                        load_hard_btn.click(load_hard, outputs=[hard_mods, hard_outcomes, hard_just, hard_think])

                step_btn = gr.Button("Execute Strategic Step", variant="primary")

        # Logic
        def sync_from_mode(mode):
            t_id = "task_easy"
            if mode == "propose_new_rule": t_id = "task_medium"
            elif mode == "evolve_policy": t_id = "task_hard"
            
            # Perform reset with the new task_id
            res = handle_reset(t_id)
            return (t_id,) + res

        def sync_from_tab(evt: gr.SelectData):
            t_id = "task_easy"
            mode = "propose_clarification"
            if evt.index == 1: 
                t_id = "task_medium"
                mode = "propose_new_rule"
            elif evt.index == 2: 
                t_id = "task_hard"
                mode = "evolve_policy"
            
            res = handle_reset(t_id)
            return (t_id, mode) + res

        # Event Listeners
        reset_btn.click(handle_reset, inputs=[task_id], outputs=[corpus_table, policy_display, best_score_disp, steps_left_disp, episode_disp, corpus_count_disp, reward_plot, reward_outcome_disp, raw_json_box])
        
        # Automatic Sync: Radio -> Dropdown & Initialize
        action_mode.change(
            sync_from_mode, 
            inputs=[action_mode], 
            outputs=[task_id, corpus_table, policy_display, best_score_disp, steps_left_disp, episode_disp, corpus_count_disp, reward_plot, reward_outcome_disp, raw_json_box]
        )
        
        # Automatic Sync: Tab -> Dropdown & Radio & Initialize
        action_tabs.select(
            sync_from_tab,
            outputs=[task_id, action_mode, corpus_table, policy_display, best_score_disp, steps_left_disp, episode_disp, corpus_count_disp, reward_plot, reward_outcome_disp, raw_json_box]
        )

        step_btn.click(
            handle_step,
            inputs=[
                task_id, action_mode,
                easy_term, easy_def, easy_just, easy_think,
                med_domain, med_rule, med_scope, med_just, med_think,
                hard_mods, hard_outcomes, hard_just, hard_think
            ],
            outputs=[corpus_table, policy_display, best_score_disp, steps_left_disp, episode_disp, corpus_count_disp, reward_plot, reward_outcome_disp, raw_json_box]
        )

    return demo

# Enable the professional "Judge Ready" console by default
custom_demo = build_custom_ui()
app = gr.mount_gradio_app(app, custom_demo, path="/dashboard/")

def main():
    # Sync with Hugging Face and local requirements
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
