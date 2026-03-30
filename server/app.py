# server/app.py
from __future__ import annotations
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, RedirectResponse
import uvicorn
from openenv.core.env_server import create_fastapi_app
from models import (
    ProposeClarificationAction, ProposeNewRuleAction, EvolveProcessAction,
    Observation, Action
)
from server.environment import PolicyEvolverEnvironment
from server.grader import grade
from server.tasks import TASK_REGISTRY
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


@app.get("/")
async def root():
    return {
        "status": "healthy", 
        "message": "PolicyEvolverEnv is running perfectly. Please append /docs to your URL to view the interactive API.",
        "endpoints": ["/health", "/tasks", "/step", "/reset", "/state", "/grader", "/baseline"]
    }


@app.get("/web")
async def web():
    """HF Spaces pings this path to confirm the Docker app is alive."""
    from fastapi.responses import HTMLResponse
    return HTMLResponse("""
    <html><head><title>PolicyEvolverEnv</title></head>
    <body style="font-family:system-ui;background:#0d1117;color:#c9d1d9;display:flex;justify-content:center;align-items:center;height:100vh;margin:0">
    <div style="text-align:center;max-width:600px">
        <h1 style="color:#58a6ff">PolicyEvolverEnv</h1>
        <p>RL environment for policy evolution through meta-reasoning</p>
        <p style="margin-top:20px"><a href="/docs" style="color:#58a6ff;text-decoration:none;border:1px solid #58a6ff;padding:8px 16px;border-radius:6px">Open API Docs →</a></p>
        <p style="color:#8b949e;font-size:0.85em;margin-top:30px">Endpoints: /reset · /step · /state · /health · /tasks · /baseline</p>
    </div></body></html>
    """)


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
    from inference import run_direct_baseline
    results = await run_direct_baseline()
    return results


def main():
    """Entry point for the OpenEnv multi-mode deployment grader."""
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    main()
