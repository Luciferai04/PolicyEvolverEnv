---
title: PolicyEvolverEnv
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
base_path: /dashboard/
---
#  PolicyEvolverEnv — Multi-Modal Strategic Governance Sandbox

**PolicyEvolverEnv** is an OpenEnv-compliant reinforcement learning environment designed for the **Meta × PyTorch × Scaler Hackathon**. It serves as a production-grade benchmark for demonstrating in-context policy improvement using RLVR signals — no weight updates required, making the environment compute-efficient and immediately deployable.

---

### 🧪 Advanced Reward Shaping (RLVR Integration)
Unlike standard environments with static rewards, **PolicyEvolverEnv v2.0** implements a sophisticated, deterministic grading engine designed to harden LLM strategic reasoning:

*   **Tiered CoT Bonus**: Rewards analytical reasoning (up to +0.20) based on keyword density and length.
*   **Clarity Coherence**: Penalizes "vague" or "subjective" (e.g., *maybe, perhaps*) policy definitions in Easy tasks.
*   **Tradeoff Realism**: Detects and caps "hallucinated" outcomes in Hard tasks (e.g., claiming to simultaneously maximize fraud-prevention and revenue).
*   **Step-Delta Shaping**: Provides an `improvement_bonus` for iterative actions that significantly outperform the episode's best score.
*   **Anti-Repetition Penalty**: Encounters a -0.30 penalty for exact repeated actions, forcing the agent toward continuous evolution.

---

##  Environment Description & Motivation
PolicyEvolverEnv is a real-world governance sandbox where an AI agent improves its in-context policy to **design and evolve governance policies** through meta-reasoning over real-world operational data. In modern platforms (social media, enterprise HR, e-commerce), static policies quickly become outdated or vaguely applied, leading to inconsistent enforcement, false-positive moderation, and unrecognized fraud. 

This environment simulates this challenge by presenting the agent with a corpus of operational data alongside an existing policy framework. The agent's goal is to analyze the outcomes, identify systemic flaws or ambiguities, and act directly on the policies to optimize governance outcomes. This directly tackles live production problems faced by platforms like Meta.

##  The Strategic Concept

### 1. The Core Idea: What is PolicyEvolverEnv?
Most AI environments are games (like Chess or Atari). **PolicyEvolverEnv** is different—it is a **Strategic Governance Sandbox**.

The environment represents the **Reinforcement Learning from Verifiable Rewards (RLVR)** stage of inference-time adaptation. It gives an agent a score (Reward) based on how well it identifies a flaw in a policy and "evolves" it to be more precise.

*   **The Problem**: Human moderators or automated systems make mistakes because the "Rules of the Game" are broken.
*   **The Solution**: An AI agent that doesn't just follow rules, but **designs** them.

### 2. The Gradio "Judge Console": How it Works
The dashboard we built (`server/app.py`) is the human-readable window into this environment. It’s designed as a **Command & Control** center for a "Policy Judge."

####  The Left Panel: Scenario Metrics
*   **Environment Best Score**: This tracks the highest score achieved in this session. It represents the "Gold Standard" the agent is aiming for.
*   **Remaining Execution Steps**: Each "Episode" has a limit (5 steps). The agent must improve the policy within this budget. This forces **Strategic Efficiency**.
*   **Latest Strategic Reward**: Every time you click "Execute," the Grader (`server/grader.py`) analyzes your proposal. If it’s vague, you get a low reward (0.1–0.3). If it’s specific and measurable, you get a high reward (0.8–0.9).

####  The Right Panel: Observations
*   **Data Corpus (Tabular View)**: These are the "Facts on the Ground." These are real-world incidents (e.g., a post flagged for 'harassment' vs one that wasn't).
*   **Active Framework**: This shows the current "Code of Law."
*   **The Workflow**: Your goal is to find an incident in the Corpus that doesn't fit correctly into the Framework, then use the bottom console to fix it.

### 3. The Power Buttons: Action Space
At the bottom, you have the **Action Console**. This is where the "Evolution" happens:

*   **Initialize Scenario**: This "boots" a specific challenge.
    *   **Easy**: Fixing vague words.
    *   **Medium**: Finding a completely missing category.
    *   **Hard**: Balancing complex trade-offs (like reducing fraud without hurting good sellers).
*   **Load Expert Suggestion**: This populates the form with a "Perfect" answer. It shows the Judge exactly what a high-performing agent looks like.
*   **Execute Strategic Step**: This is the most important button. It takes everything you typed, packages it into a Pydantic Model (`models.py`), and sends it to the environment. It triggers the **Refinement Loop**: The agent sees its score, reads the feedback, and tries again in the next step to get a higher reward.

### 4. The Final Result: Strategic Convergence
The goal of the whole idea is **Strategic Convergence**. When the "Current Project Score" hits **0.85 or higher**, it means the Agent has successfully evolved the policy framework to a point where it is:

*   **Objective**: No more biased "gut-feel" moderation.
*   **Measurable**: Success is defined by numbers (Precision/Recall).
*   **Future-Proof**: The agent has filled gaps (like AI-generated content) that didn't exist when the original rules were written.

## Observation Space
The `Observation` received by the agent at every step describes the current operational context:
- `task_id` (str): Identifier for the active scenario.
- `episode_id` (str): Unique session tracker.
- `step_count` (int): Active step number (Max 5 per episode).
- `data_corpus` (List[Dict]): Represents operational examples like social media posts, HR incidents, or seller accounts along with the action taken or outcome.
- `current_policies` (List[Dict]): The list of current active policies the system follows.
- `system_metrics` & `policy_outcomes`: Operational statistics reflecting precision/recall or false-positive rates.
- `identified_issues`: Current known flaws in the governance pipeline.

## Action Space
The Action space utilizes a highly structured Discriminated Union model to represent multi-faceted policy adjustments:

**1. ProposeClarificationAction (`propose_clarification`)** 
  - Targets an `ambiguous_term` in an existing policy.
  - Requires a specific, measurable `suggested_definition` and `justification`.
**2. ProposeNewRuleAction (`propose_new_rule`)** 
  - Addresses an unhandled domain (`rule_domain`).
  - Requires `new_rule` text, application `scope`, and `integration_points` connecting to older policies.
**3. EvolveProcessAction (`evolve_policy`)** 
  - The hardest action; holistically modifies existing rules.
  - Requires a list of `policy_modifications`, realistic `expected_outcomes` deltas, and multi-metric `rollback_conditions`.

*(Each action also supports an optional `think` property allowing Chain-of-Thought meta-reasoning for a reward score bonus).*

## Tasks
The environment provides three procedural tasks designed to ramp up in cognitive reasoning difficulty:

| Task ID | Difficulty | Expected Score | Description |
|---|---|---|---|
| `task_easy` | **Easy** | `~0.80` | **Ambiguity Clarification**: Identify and clarify vague policy terms (e.g., "harassment") in a social media community guideline to improve moderation consistency. |
| `task_medium` | **Medium** | `~0.70` | **Gap Detection**: Detect uncovered HR policy scenarios involving emerging tech (AI use, gig-worker boundaries) and propose entirely new mandatory rules. |
| `task_hard` | **Hard** | `~0.55` | **Holistic Evolution**: Analyze complex e-commerce Trust & Safety trade-offs (e.g., false-positive suspensions vs. fraud recall) to rewrite existing volume/return rate policies simultaneously. |

## Setup & Usage

### 1. Local Installation
```bash
git clone <repository_url>
cd policy_evolver_env
python3 -m venv .venv
source .venv/bin/activate
pip install -r server/requirements.txt
```

### 2. Run the Environment API
Start the FastAPI environment server locally:
```bash
uvicorn server.app:app --port 7860
```
This boots all core endpoint paths (`/reset`, `/step`, `/state`, `/tasks`, `/grader`, `/health`).

### 3. Run the Inference Baseline (Hackathon Entry)
The primary entry point for evaluation is **`inference.py`** in the root directory. This script strictly follows the Meta Hackathon `[START]`, `[STEP]`, `[END]` logging format.

Export your environment variables:
```bash
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.3-70b-versatile"
export HF_TOKEN="your_token_here"
```

Execute the baseline evaluation:
```bash
python3 inference.py
```
*(Optionally, you can run a specific task: `python3 inference.py task_easy`)*.

---

*(Note: The legacy baseline at `baseline/run_baseline.py` is still available for detailed JSON analytical reports but does not follow the hackathon logging format).*

## Baseline Performance — In-Context Policy Improvement

The agent uses **In-Context Reinforcement Learning (ICL-RL)**: no weight updates are performed. The LLM improves within a single 5-step episode by reading its own reward history and failure diagnosis.

| Task | Step 1 | Step 2 | Step 3 | Step 4 | Step 5 | Converged |
|------|--------|--------|--------|--------|--------|-----------|
| task_easy   | 0.94 | N/A  | N/A  | N/A  | N/A  | ✅ |
| task_medium | 1.00 | N/A  | N/A  | N/A  | N/A  | ✅ |
| task_hard   | 0.90 | N/A  | N/A  | N/A  | N/A  | ✅ |

**Model:** llama-3.1-8b-instant (via Groq)  
**Reproducible:** temperature=0.0, seed=42 (**Bit-for-bit identical results verified**)  
**No fine-tuning required.** The environment provides the learning signal; the model adapts its in-context policy each step.

## Setup

### Required Environment Variables

| Variable | Description | Example |
|---|---|---|
| HF_TOKEN | API key for LLM inference (Groq) | gsk_... |
| API_BASE_URL | Provider endpoint | https://api.groq.com/openai/v1 |
| MODEL_NAME | Model identifier | llama-3.1-8b-instant |

### Getting a Free Groq API Key
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (no credit card required)
3. API Keys → Create API Key
4. Export: `export HF_TOKEN=gsk_your_key_here`

## 📈 Strategic Reward Evolution & RLVR
PolicyEvolverEnv serves as the **Strategic Sandbox** for the **Reinforcement Learning from Verifiable Rewards (RLVR)** stage of the modern LLM inference pipeline. Unlike static evaluation, this environment enables agents to refine their strategies iteratively based on high-quality, verifiable feedback.

![Reward Progression](https://raw.githubusercontent.com/Luciferai04/PolicyEvolverEnv/master/reward_progression.png)

### 🧠 How It Works: The Iterative Refinement Process
1.  **Refinement Hub**: The baseline agent tracks its previous rewards and actions through the observation's metadata (`info`).
2.  **Strategic pivoting**: If a policy proposal receives low rewards (due to lack of specificity or missing justifications), the agent identifies the failure points and pivots its strategy in subsequent steps.
3.  **Measurable Improvement**: As shown in the progression chart, iterative refinement leads to **Strategic Convergence**, where the policy quality reaches institutional standards (Score ≥ 0.85).

For a detailed technical dive into how our project maps to RLHF/RLVR training architectures, see **[STRATEGIC_LEARNING.md](STRATEGIC_LEARNING.md)**.
