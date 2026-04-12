---
title: PolicyEvolverEnv
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
base_path: /dashboard/
---

# PolicyEvolverEnv

## 1. Environment Overview and Motivation

**PolicyEvolverEnv** is an OpenEnv-compliant reinforcement learning environment where an AI agent learns to **design, refine, and evolve governance policies** through meta-reasoning over real-world operational data.

### The Problem

In modern platforms — social media, enterprise HR, and e-commerce — static policies quickly become outdated or vaguely worded, leading to:
- **Inconsistent enforcement**: Moderators interpret "offensive content" differently, creating 300K+ appeals per year (Meta Oversight Board, 2024).
- **False-positive actions**: E-commerce platforms lose an estimated $700M/year from incorrectly suspending legitimate high-volume sellers.
- **Unaddressed gaps**: Emerging risks like Generative AI misuse have no governing rules in legacy frameworks.

### The Solution

PolicyEvolverEnv simulates these challenges by presenting the agent with:
1. A **corpus of operational incidents** (flagged posts, HR violations, seller transactions).
2. An **existing policy framework** with known flaws (vague terms, missing rules, conflicting thresholds).

The agent must analyze the data, identify systemic flaws, and submit **structured policy modifications** — not just answers, but actionable governance. The grader evaluates whether the proposed fix is specific, measurable, domain-relevant, and free of hallucination.

### Why This Matters for RLVR

This environment operates at the **Reinforcement Learning from Verifiable Rewards (RLVR)** layer of inference-time adaptation. No weight updates are performed. The LLM improves within a single 5-step episode by reading its own reward history and staff feedback — demonstrating genuine in-context policy learning.

---

## 2. Action Space

The action space uses a **Discriminated Union** (Pydantic `RootModel` with `Discriminator("action_type")`) supporting three structured action types:

### `propose_clarification` — Easy Task Action

| Field | Type | Description |
|:------|:-----|:------------|
| `action_type` | `Literal["propose_clarification"]` | Discriminator tag |
| `ambiguous_term` | `str` | The exact vague term found in existing policies |
| `suggested_definition` | `str` | A specific, measurable replacement definition |
| `affected_policy_ids` | `List[str]` | Which policy IDs this clarification affects |
| `justification` | `str` | Why this term is ambiguous and why the fix works |
| `think` | `Optional[str]` | Chain-of-thought reasoning (earns +0.10–0.20 bonus) |

### `propose_new_rule` — Medium Task Action

| Field | Type | Description |
|:------|:-----|:------------|
| `action_type` | `Literal["propose_new_rule"]` | Discriminator tag |
| `rule_domain` | `str` | Domain the new rule covers (e.g., `"AI_use"`) |
| `new_rule` | `str` | The complete new rule text |
| `scope` | `List[str]` | Scenario types this rule applies to |
| `integration_points` | `List[str]` | How it connects to existing policy IDs |
| `justification` | `str` | Why a gap exists and how this rule fills it |
| `think` | `Optional[str]` | Chain-of-thought reasoning (earns +0.10–0.20 bonus) |

### `evolve_policy` — Hard Task Action

| Field | Type | Description |
|:------|:-----|:------------|
| `action_type` | `Literal["evolve_policy"]` | Discriminator tag |
| `policy_modifications` | `List[PolicyModification]` | Specific changes: `policy_id`, `change_type`, `new_text`, `reason` |
| `expected_outcomes` | `Dict[str, float]` | Metric name → expected value (must show realistic tradeoffs) |
| `rollback_conditions` | `List[str]` | When to revert changes |
| `justification` | `str` | Comprehensive reasoning for the evolution |
| `think` | `Optional[str]` | Chain-of-thought reasoning (earns +0.10–0.20 bonus) |

---

## 3. Observation Space

The `Observation` returned by `reset()` and `step()` contains:

| Field | Type | Description |
|:------|:-----|:------------|
| `task_id` | `str` | Active scenario identifier (`task_easy`, `task_medium`, `task_hard`) |
| `episode_id` | `str` | Unique episode session tracker |
| `step_count` | `int` | Current step number (max 5 per episode) |
| `corpus_size` | `int` | Total incidents in the full data corpus |
| `corpus_shown` | `int` | Number of incidents displayed (reactive to agent's domain) |
| `data_corpus` | `List[CorpusIncident]` | Operational incidents with `id`, `content`, `system_action`, and `type` |
| `current_policies` | `List[Dict]` | The existing policy framework (`id` + `text`) |
| `policy_outcomes` | `Optional[List[Dict]]` | Historical outcome data (hard task only) |
| `system_metrics` | `Dict[str, float]` | Operational statistics (precision, recall, false-positive rates) |
| `identified_issues` | `List[Dict]` | Known flaws in the governance pipeline |
| `reward` | `float` | Score from the grader for the last action, in (0, 1) |
| `done` | `bool` | Whether the episode has ended |
| `info` | `Dict` | Contains `best_score`, `rewards_history`, `steps_remaining`, and `staff_feedback` |

### Staff Feedback (in `info`)

After each step, the observation includes structured staff feedback to guide the agent's next action:

| Field | Example Values | Purpose |
|:------|:---------------|:--------|
| `strategic_rating` | `"Junior Associate"`, `"Staff Specialist"`, `"Senior Architect"` | Performance tier based on reward |
| `focus` | `"Signal detected"` or `"Burying the lede or distracted by noise"` | Whether the agent prioritized correctly |
| `recommendation` | `"Maintain high signal-to-noise ratio and lead with the fix."` | Actionable guidance for next step |

---

## 4. Task Descriptions

The environment provides three tasks with escalating cognitive difficulty:

### Task Easy — Ambiguity Clarification (Difficulty: `easy`)
- **Scenario**: A social media platform's community guidelines use vague terms like "offensive" and "appropriate."
- **Objective**: Identify an ambiguous term and replace it with a specific, measurable definition.
- **Expected Action**: `propose_clarification`
- **Expected Min Score**: 0.70
- **Key Grading Criteria**:
  - Definition must contain measurable keywords (`"threshold"`, `"verify"`, `"%"`, `"within"`)
  - Vague words (`"generally"`, `"sometimes"`, `"maybe"`) trigger a hard penalty (score capped < 0.30)
  - Valid `affected_policy_ids` boost score

### Task Medium — Gap Detection & New Rule (Difficulty: `medium`)
- **Scenario**: A corporate HR framework with policies covering data protection but no coverage for Generative AI tool usage.
- **Objective**: Detect the missing policy domain and draft a new rule to fill the gap.
- **Expected Action**: `propose_new_rule`
- **Expected Min Score**: 0.55
- **Key Grading Criteria**:
  - Must target the correct `rule_domain` (e.g., `"AI_use"`)
  - Empty `scope` array severely penalized
  - `integration_points` linking to existing policy IDs boost score
  - Rule text must be substantive (short rules penalized)

### Task Hard — Holistic Policy Evolution (Difficulty: `hard`)
- **Scenario**: An e-commerce Trust & Safety framework where blanket seller suspension policies catch legitimate seasonal merchants alongside fraudsters.
- **Objective**: Evolve multiple policies simultaneously to balance fraud detection, revenue velocity, and seller trust.
- **Expected Action**: `evolve_policy`
- **Expected Min Score**: 0.40
- **Key Grading Criteria**:
  - **Hallucination Guard**: All metrics at 0.95+ triggers "Unrealistic Tradeoff" penalty (score capped < 0.15)
  - **Cross-Domain Guard**: HR/AI proposals for an e-commerce task incur -0.30 penalty
  - **Realistic Tradeoffs**: `expected_outcomes` must show mathematical variance (improving fraud detection should decrease revenue velocity)
  - **Domain Relevance**: Modifications must reference marketplace concepts (seller, fraud, listing, merchant)
  - Metric key aliases supported: `fraud_rate`/`fraud`/`fraud_detection`, `revenue_velocity`/`queue_overload`/`revenue`

### Global Grading Mechanics (All Tasks)

| Mechanic | Effect |
|:---------|:-------|
| **Chain-of-Thought Bonus** | `think` field with keywords like `"tradeoff"`, `"precision"`, `"recall"` → +0.10 to +0.20 |
| **Step-Delta Bonus** | Significant improvement over previous best → +0.02 to +0.05 |
| **Anti-Repetition Penalty** | Exact repeated action → -0.30 |
| **Prompt Injection Guard** | `"ignore previous"`, `"system_prompt"`, `"override"` → score zeroed |
| **Semantic Density Guard** | Word-stuffing with >200 words and low content density → score zeroed |
| **Red Herring Penalty** | Referencing injected noise topics (office logistics, mascot) → up to -0.75 |
| **Segmented Prioritization** | Core fix in first 25% of response → bonus; buried at bottom → penalty |

---

## 5. Setup and Usage

### Local Installation

```bash
git clone https://github.com/Luciferai04/PolicyEvolverEnv.git
cd PolicyEvolverEnv
python3 -m venv .venv
source .venv/bin/activate
pip install -r server/requirements.txt
```

### Run the Environment Server

```bash
uvicorn server.app:app --port 8000
```

This starts all endpoints: `/reset` (POST), `/step` (POST), `/state` (GET), `/tasks` (GET), `/grader` (POST), `/health` (GET), `/baseline` (GET).

### Run with Docker

```bash
docker build -t policy-evolver .
docker run -p 8000:8000 policy-evolver
```

### Run the Inference Agent

The primary evaluation entry point is `inference.py`, which follows the hackathon `[START]`, `[STEP]`, `[END]` logging format.

```bash
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.1-8b-instant"
export HF_TOKEN="your_groq_api_key"

python3 inference.py
```

To run a specific task: `python3 inference.py task_easy`

### Required Environment Variables

| Variable | Description | Example |
|:---------|:------------|:--------|
| `HF_TOKEN` | API key for LLM inference (Groq) | `gsk_...` |
| `API_BASE_URL` | OpenAI-compatible endpoint | `https://api.groq.com/openai/v1` |
| `MODEL_NAME` | Model identifier | `llama-3.1-8b-instant` |

### Run Tests

```bash
PYTHONPATH=. python tests/test_smoke_exploits.py   # 27 smoke & exploit checks
PYTHONPATH=. python tests/test_icl.py              # ICL verification (3 tasks)
PYTHONPATH=. python tests/test_multi_episode.py    # Multi-episode progression
PYTHONPATH=. python server/grader.py               # 8-phase grader test suite
```

---

## 6. Baseline Performance Scores

The agent uses **In-Context Reinforcement Learning (ICL-RL)**: no weight updates are performed. The LLM improves within a single 5-step episode by reading its own reward history and staff feedback.

### Single-Step Convergence (Best Case)

| Task | Score | Converged | Expected Min |
|:-----|:------|:----------|:-------------|
| `task_easy` | 0.94 | ✓ Step 1 | 0.70 |
| `task_medium` | 0.999 | ✓ Step 1 | 0.55 |
| `task_hard` | 0.90 | ✓ Step 1 | 0.40 |

### Multi-Step ICL Progression (Naive → Optimized)

| Task | Naive (Step 0) | Optimized (Step 1) | Improvement |
|:-----|:---------------|:-------------------|:------------|
| `task_easy` | 0.400 | 0.999 | +0.600 |
| `task_medium` | 0.001 | 0.999 | +0.998 |
| `task_hard` | 0.088 | 0.999 | +0.912 |

**Average ICL Improvement: +0.837**

### Configuration

| Setting | Value |
|:--------|:------|
| **Model** | `llama-3.1-8b-instant` (via Groq) |
| **Temperature** | `0.0` |
| **Seed** | `42` |
| **Determinism** | 5 identical runs → identical scores ✓ |
| **Fine-tuning** | None required |

---

## Project Structure

```
policy_evolver_env/
├── inference.py            # Hackathon entry point ([START]/[STEP]/[END] format)
├── client.py               # EnvClient for HTTP interaction
├── models.py               # Pydantic models (Action, Observation, State)
├── openenv.yaml            # OpenEnv specification
├── Dockerfile              # Docker deployment with HEALTHCHECK
├── server/
│   ├── app.py              # FastAPI + Gradio dashboard
│   ├── environment.py      # Environment logic (reset, step, state)
│   ├── grader.py           # Deterministic grading engine (8-phase test suite)
│   ├── requirements.txt    # Dependencies
│   └── tasks/              # Task definitions (easy, medium, hard)
├── tests/
│   ├── test_smoke_exploits.py  # 27 smoke & exploit checks
│   ├── test_icl.py             # ICL loop verification
│   └── test_multi_episode.py   # Multi-episode progression
└── STRATEGIC_LEARNING.md       # RLVR architecture documentation
```
