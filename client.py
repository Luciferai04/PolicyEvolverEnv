# client.py
from openenv.core.env_client import EnvClient
from .models import Action, Observation, State


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
