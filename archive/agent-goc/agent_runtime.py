from __future__ import annotations

import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Personal Agent Runtime", version="0.1.0")

WORKSPACE = Path(os.getenv("AGENT_WORKSPACE", "./workspace")).resolve()
AGENT_TOKEN = os.getenv("AGENT_TOKEN")
MAX_OUTPUT = int(os.getenv("AGENT_MAX_OUTPUT", "12000"))
TIMEOUT = int(os.getenv("AGENT_COMMAND_TIMEOUT", "120"))

# Conservative allow-list for v0.1. Expand deliberately as tools are added.
ALLOWED_PREFIXES = (
    "pwd", "ls", "find", "cat", "head", "tail", "grep", "git status",
    "git log", "python --version", "python3 --version", "node --version",
    "npm --version", "uname", "whoami", "date",
)

WORKSPACE.mkdir(parents=True, exist_ok=True)


class TaskRequest(BaseModel):
    command: str = Field(min_length=1, max_length=2000)
    timeout: int | None = Field(default=None, ge=1, le=600)


class TaskResult(BaseModel):
    job_id: str
    status: Literal["completed", "failed"]
    started_at: str
    finished_at: str
    exit_code: int | None
    stdout: str
    stderr: str


def require_token(authorization: str | None) -> None:
    if not AGENT_TOKEN:
        raise HTTPException(503, "AGENT_TOKEN is not configured")
    if authorization != f"Bearer {AGENT_TOKEN}":
        raise HTTPException(401, "Invalid agent token")


def allowed(command: str) -> bool:
    normalized = " ".join(command.strip().split())
    return any(normalized == prefix or normalized.startswith(prefix + " ") for prefix in ALLOWED_PREFIXES)


@app.get("/health")
def health() -> dict[str, object]:
    return {"ok": True, "version": app.version, "workspace": str(WORKSPACE)}


@app.post("/v1/execute", response_model=TaskResult)
def execute(req: TaskRequest, authorization: str | None = Header(default=None)) -> TaskResult:
    require_token(authorization)
    if not allowed(req.command):
        raise HTTPException(403, "Command is outside the v0.1 allow-list")

    job_id = str(uuid.uuid4())
    started = datetime.now(timezone.utc).isoformat()
    try:
        p = subprocess.run(
            req.command,
            cwd=WORKSPACE,
            shell=True,
            capture_output=True,
            text=True,
            timeout=req.timeout or TIMEOUT,
        )
        status = "completed" if p.returncode == 0 else "failed"
        exit_code = p.returncode
        stdout = p.stdout[-MAX_OUTPUT:]
        stderr = p.stderr[-MAX_OUTPUT:]
    except subprocess.TimeoutExpired as exc:
        status = "failed"
        exit_code = None
        stdout = (exc.stdout or "")[-MAX_OUTPUT:] if isinstance(exc.stdout, str) else ""
        stderr = "Command timed out"
    finished = datetime.now(timezone.utc).isoformat()
    return TaskResult(
        job_id=job_id,
        status=status,
        started_at=started,
        finished_at=finished,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
    )
