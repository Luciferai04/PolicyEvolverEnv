# client.py
from typing import Any, Dict
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from models import Action, Observation, State


class PolicyEvolverEnv(EnvClient):
    """
    Client for PolicyEvolverEnv.
    Usage:
        async with PolicyEvolverEnv(base_url="https://your-space.hf.space") as env:
            obs = await env.reset(task_id="task_easy")
            result = await env.step(action)
    """
    observation_class = Observation
    state_class = State

    def _step_payload(self, action: Any) -> Dict[str, Any]:
        """Convert an Action to the JSON payload expected by the server."""
        if isinstance(action, dict):
            return action
        if hasattr(action, "model_dump"):
            return action.model_dump()
        if hasattr(action, "__dict__"):
            return vars(action)
        return dict(action)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult:
        """Convert server JSON response to a StepResult."""
        obs_data = payload.get("observation", {})
        return StepResult(
            observation=obs_data,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Parse state response from the server."""
        return payload
