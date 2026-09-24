"""Hàng đợi job trong bộ nhớ (từ `gateway.py` của repo Agent), không phụ thuộc web framework.

Khác bản gốc: giới hạn số job giữ trong bộ nhớ, và chỉ job đang chạy ("running") mới nhận kết quả.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

JobStatus = Literal["queued", "running", "completed", "failed"]
MAX_INSTRUCTION = 10_000


class JobError(Exception):
    """Lỗi thao tác job; `status` là mã HTTP tương ứng (404: không có job, 409: sai trạng thái, 429: hàng đợi đầy)."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


@dataclass
class Job:
    id: str
    instruction: str
    status: JobStatus = "queued"
    result: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {"job_id": self.id, **{key: value for key, value in asdict(self).items() if key != "id"}}


class JobQueue:
    def __init__(self, max_jobs: int = 1000):
        self._jobs: dict[str, Job] = {}; self._lock = threading.Lock(); self.max_jobs = max_jobs

    def __len__(self) -> int:
        return len(self._jobs)

    def create(self, instruction: str) -> Job:
        if not isinstance(instruction, str) or not 1 <= len(instruction) <= MAX_INSTRUCTION: raise ValueError(f"Yêu cầu phải là chuỗi dài 1–{MAX_INSTRUCTION} ký tự")
        with self._lock:
            if len(self._jobs) >= self.max_jobs:
                finished = [job for job in self._jobs.values() if job.status in ("completed", "failed")]
                if not finished: raise JobError(429, f"Hàng đợi đã đầy ({self.max_jobs} job chưa xong)")
                del self._jobs[min(finished, key=lambda job: job.created_at).id]  # bỏ job đã xong cũ nhất
            job = Job(id=str(uuid.uuid4()), instruction=instruction); self._jobs[job.id] = job
            return job

    def get(self, job_id: str) -> Job:
        with self._lock: return self._find(job_id)

    def claim(self, job_id: str) -> Job:
        """Máy chạy nhận một job đang chờ; job chuyển sang "running"."""
        with self._lock:
            job = self._find(job_id)
            if job.status != "queued": raise JobError(409, f"Job đang ở trạng thái {job.status}, không nhận được")
            job.status = "running"; return job

    def submit_result(self, job_id: str, payload: dict[str, Any]) -> Job:
        """Trả kết quả cho job đang chạy: `ok` là True thì "completed", ngược lại "failed"."""
        if not isinstance(payload, dict): raise ValueError("Kết quả phải là một object JSON")
        with self._lock:
            job = self._find(job_id)
            if job.status != "running": raise JobError(409, f"Job đang ở trạng thái {job.status}, chưa được nhận nên không trả kết quả được")
            job.result = payload; job.status = "completed" if payload.get("ok") is True else "failed"; return job

    def _find(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)
        if job is None: raise JobError(404, "Không tìm thấy job")
        return job
