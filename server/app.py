# server/app.py
from __future__ import annotations
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from openenv.core.env_server import create_fastapi_app
from ..models import (
    ProposeClarificationAction, ProposeNewRuleAction, EvolveProcessAction,
    Observation, Action
)
from .environment import PolicyEvolverEnvironment
from .grader import grade
from .tasks import TASK_REGISTRY
import json

# Create app via OpenEnv helper — pass factory callable, action/obs classes
app = create_fastapi_app(
    env=PolicyEvolverEnvironment,
    action_cls=Action,          # Pydantic union
    observation_cls=Observation,
)


@app.get("/health")
async def health():
    return {"status": "healthy", "environment": "PolicyEvolverEnv", "version": "1.0.0"}


@app.get("/tasks")
async def list_tasks():
    return [
        {
            "task_id": tid,
            "difficulty": t["difficulty"],
            "description": t["description"],
            "num_policies": t["num_policies"],
            "num_data_points": t["num_data_points"],
        }
        for tid, t in TASK_REGISTRY.items()
    ]


@app.get("/grader")
async def grader_endpoint(
    task_id: str = Query(..., description="task_easy | task_medium | task_hard"),
    action_json: str = Query(..., description="JSON-encoded action dict"),
):
    try:
        action_dict = json.loads(action_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="action_json must be valid JSON")
    score = grade(action_dict, task_id)
    return {"task_id": task_id, "score": score}


@app.get("/baseline")
async def run_baseline_endpoint():
    """
    Runs the LLM baseline (or rule-based fallback) and returns scores for all tasks.
    Uses the grader directly instead of HTTP calls to self.
    """
    from ..inference import run_direct_baseline
    results = await run_direct_baseline()
    return results
