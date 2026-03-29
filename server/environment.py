# server/environment.py
from __future__ import annotations
import uuid
import random
from typing import Optional, Any, Dict
from openenv.core.env_server import Environment
from ..models import (
    Action, Observation, State,
    ProposeClarificationAction, ProposeNewRuleAction, EvolveProcessAction,
)
from .grader import grade
from .tasks import TASK_REGISTRY


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

        task = TASK_REGISTRY[task_id]
        self._current_task = task
        self._state = State(
            episode_id=episode_id or str(uuid.uuid4()),
            task_id=task_id,
            step_count=0,
            max_steps=5,
            current_score=0.0,
            best_score=0.0,
            actions_taken=[],
        )

        return Observation(
            task_id=task_id,
            episode_id=self._state.episode_id,
            step_count=0,
            data_corpus=task["data_corpus"],
            current_policies=task["current_policies"],
            policy_outcomes=task.get("policy_outcomes"),
            system_metrics=task.get("system_metrics", {}),
            identified_issues=task.get("identified_issues", []),
            reward=0.0,
            done=False,
            info={"task_description": task["description"], "difficulty": task["difficulty"]},
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        if self._current_task is None:
            raise RuntimeError("Call reset() before step()")

        self._state.step_count += 1

        # action can be a dict from the API or a Pydantic model
        if isinstance(action, dict):
            action_dict = action
        else:
            action_dict = action.model_dump() if hasattr(action, "model_dump") else dict(action)

        reward = grade(action_dict, self._state.task_id)
        self._state.current_score = reward
        self._state.best_score = max(self._state.best_score, reward)

        action_type = action_dict.get("action_type", "unknown") if isinstance(action_dict, dict) else "unknown"
        self._state.actions_taken.append(action_type)

        done = (
            reward >= 0.90 or
            self._state.step_count >= self._state.max_steps
        )

        return Observation(
            task_id=self._state.task_id,
            episode_id=self._state.episode_id,
            step_count=self._state.step_count,
            data_corpus=self._current_task["data_corpus"],
            current_policies=self._current_task["current_policies"],
            policy_outcomes=self._current_task.get("policy_outcomes"),
            system_metrics=self._current_task.get("system_metrics", {}),
            identified_issues=self._current_task.get("identified_issues", []),
            reward=reward,
            done=done,
            info={
                "best_score": self._state.best_score,
                "steps_remaining": self._state.max_steps - self._state.step_count,
            },
        )

    @property
    def state(self) -> State:
        return self._state
