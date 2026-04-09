from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.environment import DataCleaningEnv
from env.models import Action, Observation, StepResult
from env.tasks import list_tasks as list_task_specs

app = FastAPI(
    title="Data Cleaning OpenEnv Benchmark",
    version="1.0.0",
    description="LLM agent benchmark for real-world data cleaning tasks.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: Dict[str, DataCleaningEnv] = {}


@app.get("/")
def root():
    tasks = list_task_specs()
    return {
        "name": "Data Cleaning OpenEnv Benchmark",
        "version": "1.0.0",
        "tasks": tasks,
        "api": {
            "reset": "POST /reset",
            "step": "POST /step/{session_id}",
            "step_compat": "POST /step",
            "state": "GET  /state/{session_id}",
            "state_compat": "GET  /state?session_id=...",
            "metadata": "GET  /metadata",
            "schema": "GET  /schema",
            "mcp": "GET|POST /mcp",
            "health": "GET  /health",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok", "sessions_active": len(sessions)}


class ResetRequest(BaseModel):
    task_id: Optional[str] = None


@app.post("/reset")
def reset(body: ResetRequest = ResetRequest()):
    session_id = str(uuid.uuid4())
    env = DataCleaningEnv()
    obs = env.reset(task_id=body.task_id)
    sessions[session_id] = env
    return {
        "session_id": session_id,
        "observation": obs.model_dump(),
        "reward": 0.0,
        "done": False,
        "info": {
            "error": None,
            "cumulative_reward": env.cumulative_reward,
            "raw_cumulative_reward": env.raw_cumulative_reward,
            "final_score": env.final_score,
            "step": env.step_count,
        },
    }


@app.post("/step")
def step_compat(
    payload: Dict[str, Any],
    session_id: Optional[str] = Query(default=None),
):
    payload_session_id = payload.get("session_id")
    resolved_session_id = _resolve_session_id(payload_session_id or session_id)
    action_payload = payload.get("action", payload)

    if not isinstance(action_payload, dict):
        raise HTTPException(status_code=400, detail="Action payload must be an object")
    if "type" not in action_payload:
        raise HTTPException(status_code=400, detail="Action payload requires 'type'")

    action = Action(**action_payload)
    env = _get_session(resolved_session_id)
    result = env.step(action)
    return result.model_dump()


@app.post("/step/{session_id}")
def step(session_id: str, action: Action):
    env = _get_session(session_id)
    result = env.step(action)
    return result.model_dump()


@app.get("/state")
def state_compat(session_id: Optional[str] = Query(default=None)):
    env = _get_session(_resolve_session_id(session_id))
    return env.state()


@app.get("/state/{session_id}")
def state(session_id: str):
    env = _get_session(session_id)
    return env.state()


@app.get("/metadata")
def metadata():
    return {
        "name": "data-cleaning-benchmark",
        "version": "1.0.0",
        "description": "LLM agent benchmark for real-world data cleaning tasks.",
        "tasks": list_task_specs(),
        "score_range": {
            "min": DataCleaningEnv.MIN_EPISODE_SCORE,
            "max": DataCleaningEnv.MAX_EPISODE_SCORE,
        },
        "entrypoints": {
            "reset": "/reset",
            "step": "/step",
            "state": "/state",
            "health": "/health",
            "tasks": "/tasks",
            "schema": "/schema",
            "mcp": "/mcp",
        },
    }


@app.get("/schema")
def schema():
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "step_result": StepResult.model_json_schema(),
        "reset_request": ResetRequest.model_json_schema(),
    }


@app.api_route("/mcp", methods=["GET", "POST"])
def mcp_metadata():
    return {
        "supported": False,
        "message": "This benchmark exposes simulation HTTP endpoints (reset/step/state).",
    }


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    sessions.pop(session_id, None)
    return {"deleted": session_id}


@app.get("/tasks")
def list_tasks():
    return {"tasks": list_task_specs()}


def _resolve_session_id(session_id: Optional[str]) -> str:
    if session_id:
        return session_id
    if len(sessions) == 1:
        return next(iter(sessions.keys()))
    raise HTTPException(
        status_code=400,
        detail="session_id is required when there is not exactly one active session",
    )


def _get_session(session_id: str) -> DataCleaningEnv:
    env = sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return env
