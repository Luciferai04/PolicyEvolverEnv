---
title: PolicyEvolverEnv
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
---
# PolicyEvolverEnv

**PolicyEvolverEnv** is an OpenEnv-compliant reinforcement learning environment designed for the Meta × PyTorch × Scaler Hackathon.

## Environment Description & Motivation
PolicyEvolverEnv is a real-world governance sandbox where an AI agent learns to **design and evolve governance policies** through meta-reasoning over real-world operational data. In modern platforms (social media, enterprise HR, e-commerce), static policies quickly become outdated or vaguely applied, leading to inconsistent enforcement, false-positive moderation, and unrecognized fraud. 

This environment simulates this challenge by presenting the agent with a corpus of operational data alongside an existing policy framework. The agent's goal is to analyze the outcomes, identify systemic flaws or ambiguities, and act directly on the policies to optimize governance outcomes. This directly tackles live production problems faced by platforms like Meta.

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
uvicorn server.app:app --port 8000
```
This boots all core endpoint paths (`/reset`, `/step`, `/state`, `/tasks`, `/grader`, `/health`).

### 3. Run the Inference Baseline
The environment includes a built-in testing script named `inference.py` ready for deployment on Hugging Face Spaces.

Export your environment variables:
```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="your_huggingface_or_openai_api_key_here"
export OPENENV_BASE_URL="http://localhost:8000"
```

Execute the agent simulation against the running environment:
```bash
python inference.py --mode llm --output json
```
*(If no API key is specified, `--mode rule` will execute the deterministic rule-based fallback).*

## Baseline Scores
The bundled deterministic fallback strategy (`inference.py --mode rule`) yields the following baseline validation scores across the active grader:

- **Easy (Ambiguity Clarification):** 1.000
- **Medium (New Rule Proposal):** 1.000
- **Hard (Policy Evolution):** 0.950
- **Overall Average:** 0.983

*(Note: Live LLM runs generally average expected heuristic bounds around ~0.80, ~0.70, and ~0.55 respectively).*
