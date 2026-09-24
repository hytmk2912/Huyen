from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Personal Agent Gateway", version="0.1.0")
GATEWAY_TOKEN = os.getenv("GATEWAY_TOKEN")

jobs: dict[str, "Job"] = {}
lock = threading.Lock()


@dataclass
class Job:
    id: str
    instruction: str
    status: Literal["queued", "running", "completed", "failed"] = "queued"
    result: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class JobRequest(BaseModel):
    instruction: str = Field(min_length=1, max_length=10000)


def auth(token: str | None) -> None:
    if not GATEWAY_TOKEN:
        raise HTTPException(503, "GATEWAY_TOKEN is not configured")
    if token != GATEWAY_TOKEN:
        raise HTTPException(401, "Invalid gateway token")


@app.get("/health")
def health() -> dict[str, object]:
    return {"ok": True, "version": app.version, "jobs": len(jobs)}


@app.post("/v1/jobs")
def create_job(req: JobRequest, authorization: str | None = Header(default=None)) -> dict[str, str]:
    auth(authorization)
    job = Job(id=str(uuid.uuid4()), instruction=req.instruction)
    with lock:
        jobs[job.id] = job
    return {"job_id": job.id, "status": job.status}


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str, authorization: str | None = Header(default=None)) -> dict:
    auth(authorization)
    with lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return {
            "job_id": job.id,
            "instruction": job.instruction,
            "status": job.status,
            "result": job.result,
            "created_at": job.created_at,
        }


@app.post("/v1/jobs/{job_id}/claim")
def claim_job(job_id: str, authorization: str | None = Header(default=None)) -> dict:
    auth(authorization)
    with lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        if job.status != "queued":
            raise HTTPException(409, f"Job is {job.status}")
        job.status = "running"
        return {"job_id": job.id, "instruction": job.instruction, "status": job.status}


@app.post("/v1/jobs/{job_id}/result")
def submit_result(job_id: str, payload: dict, authorization: str | None = Header(default=None)) -> dict:
    auth(authorization)
    with lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        job.result = payload
        job.status = "completed" if payload.get("ok", False) else "failed"
        return {"job_id": job.id, "status": job.status}
