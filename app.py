from __future__ import annotations

import uuid
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.environment import DataCleaningEnv
from env.models import Action
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
            "state": "GET  /state/{session_id}",
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
    return {"session_id": session_id, "observation": obs.model_dump()}


@app.post("/step/{session_id}")
def step(session_id: str, action: Action):
    env = _get_session(session_id)
    result = env.step(action)
    return result.model_dump()


@app.get("/state/{session_id}")
def state(session_id: str):
    env = _get_session(session_id)
    return env.state()


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    sessions.pop(session_id, None)
    return {"deleted": session_id}


@app.get("/tasks")
def list_tasks():
    return {"tasks": list_task_specs()}


def _get_session(session_id: str) -> DataCleaningEnv:
    env = sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return env
