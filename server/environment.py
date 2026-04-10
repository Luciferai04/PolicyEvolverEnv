# server/environment.py
# HF Force Rebuild: 2026-03-30T17:16:00Z
from __future__ import annotations
import uuid
import random
from typing import Optional, Any, Dict
from openenv.core.env_server import Environment
from models import (
    Action, Observation, State,
    ProposeClarificationAction, ProposeNewRuleAction, EvolveProcessAction,
)
from server.grader import grade
from server.tasks import TASK_REGISTRY


class PolicyEvolverEnvironment(Environment[Action, Observation, State]):
    """
    Real-world environment: AI agent learns to evolve governance policies
    through meta-reasoning over real-world data.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        super().__init__()
        self._state = State()
        self._current_task = None
        self._persistent_best_score = 0.0
        self._seen_action_hashes = set()
        self._initialized = True

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        task_id = kwargs.get("task_id")
        if task_id is None:
            task_id = random.choice(list(TASK_REGISTRY.keys()))

        self._seen_action_hashes = set()
        task = TASK_REGISTRY[task_id]
        self._current_task = task
        self._state = State(
            episode_id=episode_id or str(uuid.uuid4()),
            task_id=task_id,
            step_count=0,
            max_steps=5,
            current_score=0.0,
            best_score=self._persistent_best_score,
            actions_taken=[],
        )

        # Deepcopy to keep episode state
        import copy
        self._episode_corpus = copy.deepcopy(task.get("data_corpus", []))
        # Ensure all incidents follow CorpusIncident schema properly
        for item in self._episode_corpus:
            if "content" not in item:
                item["content"] = item.pop("text", None) or item.pop("desc", None) or str(item.get("flags", ""))
            if "system_action" not in item:
                item["system_action"] = "pending"

        shown_corpus = self._episode_corpus[:10]

        return Observation(
            task_id=task_id,
            episode_id=self._state.episode_id,
            step_count=0,
            corpus_size=len(self._episode_corpus),
            corpus_shown=len(shown_corpus),
            data_corpus=shown_corpus,
            current_policies=task["current_policies"],
            policy_outcomes=task.get("policy_outcomes"),
            system_metrics=task.get("system_metrics", {}),
            identified_issues=task.get("identified_issues", []),
            reward=0.0,
            done=False,
            info={
                "task_description": task["description"], 
                "difficulty": task["difficulty"],
                "best_score": self._persistent_best_score,
                "steps_remaining": self._state.max_steps
            },
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        if self._current_task is None:
            raise RuntimeError("Call reset() before step()")

        if self._state.step_count >= self._state.max_steps:
             logger.warning(f"[EXPLOIT] Step-count limit exceeded for episode {self._state.episode_id}")
             return self.reset(task_id=self._state.task_id) # Force reset if they keep pushing or return an empty observation
        
        self._state.step_count += 1

        # action can be a dict from the API or a Pydantic model
        if isinstance(action, dict):
            action_dict = action
        else:
            # Handle Pydantic RootModel used for discriminated unions
            if hasattr(action, "root"):
                action = action.root
            action_dict = action.model_dump() if hasattr(action, "model_dump") else dict(action)

        # Repetition Penalty logic
        import json as _json
        try:
            action_hash = hash(_json.dumps(action_dict, sort_keys=True, default=str))
        except Exception:
            action_hash = hash(str(action_dict))

        if action_hash in self._seen_action_hashes:
            repetition_penalty = 0.30
        else:
            repetition_penalty = 0.0
            self._seen_action_hashes.add(action_hash)

        previous_score = self._state.current_score
        raw_reward = grade(action_dict, self._state.task_id, previous_score=previous_score)
        reward = max(0.001, min(0.999, raw_reward - repetition_penalty))
        
        self._state.current_score = reward
        self._state.best_score = max(self._state.best_score, reward)
        self._persistent_best_score = max(self._persistent_best_score, reward)
        self._state.rewards_history.append(reward)

        action_type = action_dict.get("action_type", "unknown") if isinstance(action_dict, dict) else "unknown"
        self._state.actions_taken.append(action_type)

        # Reactive Corpus: Prioritize items relevant to the agent's action domain
        # This makes the world visibly react to agent choices
        target_term = action_dict.get("ambiguous_term") or action_dict.get("rule_domain") or ""
        t_term = str(target_term).lower()

        # Partition: relevant items first, then remaining
        relevant = []
        remaining = []
        for item in self._episode_corpus:
            c_type = str(item.get("type", "")).lower()
            c_text = str(item.get("content", "")).lower()

            # Update system_action based on reward (stateful corpus)
            if t_term in c_text or t_term in c_type or action_type == "evolve_policy":
                if reward >= 0.7:
                    item["system_action"] = "policy_applied"
                elif 0.3 <= reward < 0.7:
                    item["system_action"] = "flagged"

            # Sort into buckets
            if t_term and (t_term in c_text or t_term in c_type):
                relevant.append(item)
            else:
                remaining.append(item)

        # Rotate the remaining window by step count so agent sees fresh data each step
        step_offset = (self._state.step_count - 1) * 3
        rotated_remaining = remaining[step_offset:] + remaining[:step_offset]

        # Build shown corpus: relevant items first, then rotated remaining, cap at 10
        prioritized_corpus = relevant + rotated_remaining
        shown_corpus = prioritized_corpus[:10]

        done = (
            reward >= 0.90 or
            self._state.step_count >= self._state.max_steps
        )

        return Observation(
            task_id=self._state.task_id,
            episode_id=self._state.episode_id,
            step_count=self._state.step_count,
            corpus_size=len(self._episode_corpus),
            corpus_shown=len(shown_corpus),
            data_corpus=shown_corpus,
            current_policies=self._current_task["current_policies"],
            policy_outcomes=self._current_task.get("policy_outcomes"),
            system_metrics=self._current_task.get("system_metrics", {}),
            identified_issues=self._current_task.get("identified_issues", []),
            reward=reward,
            done=done,
            info={
                "best_score": self._state.best_score,
                "last_reward": reward,
                "rewards_history": self._state.rewards_history,
                "action_history": self._state.actions_taken,
                "steps_remaining": self._state.max_steps - self._state.step_count,
                "staff_feedback": {
                    "strategic_rating": "Senior Architect" if reward >= 0.85 else "Staff Specialist" if reward >= 0.65 else "Junior Associate",
                    "focus": "Signal detected" if reward >= 0.5 else "Burying the lede or distracted by noise",
                    "recommendation": "Maintain high signal-to-noise ratio and lead with the fix." if reward < 0.8 else "Excellent prioritization."
                }
            },
        )

    @property
    def state(self) -> State:
        return self._state
