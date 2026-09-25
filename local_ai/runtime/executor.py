"""Chạy một lệnh trong thư mục làm việc (từ `/v1/execute` của repo Agent).

Khác bản gốc (xem docs/GOP_AGENT.md): bản gốc chạy chuỗi lệnh bằng `shell=True` nên bị chèn lệnh
(`ls; rm -rf ~`). Ở đây lệnh chỉ nhận dạng list tham số và không bao giờ qua shell; khi quá thời gian
thì dừng cả nhóm tiến trình con. Việc chọn lệnh nào được phép chạy (allowlist) do lớp gọi quyết định.
"""
from __future__ import annotations

import os
import signal
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Sequence

DEFAULT_TIMEOUT = 120
DEFAULT_MAX_OUTPUT = 12_000


@dataclass(frozen=True)
class CommandResult:
    job_id: str
    argv: tuple[str, ...]
    status: Literal["completed", "failed"]
    started_at: str
    finished_at: str
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_command(argv: Sequence[str], workspace: str | Path, timeout: float = DEFAULT_TIMEOUT, max_output: int = DEFAULT_MAX_OUTPUT) -> CommandResult:
    """Chạy `argv` (list chuỗi) trong `workspace`; giữ tối đa `max_output` ký tự cuối của stdout/stderr."""
    if isinstance(argv, (str, bytes)) or not argv or not all(isinstance(item, str) for item in argv):
        raise TypeError("Lệnh phải là list các chuỗi (ví dụ ['ls', '-la']), không nhận chuỗi lệnh để tránh bị chèn lệnh qua shell")
    if timeout <= 0: raise ValueError("timeout phải lớn hơn 0 giây")
    directory = Path(workspace).resolve(); directory.mkdir(parents=True, exist_ok=True)
    job_id, started, command = str(uuid.uuid4()), _now(), tuple(argv)
    try:
        process = subprocess.Popen(list(command), cwd=directory, shell=False, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, encoding="utf-8", errors="replace", start_new_session=True)
    except (FileNotFoundError, PermissionError) as error:
        reason = "không tìm thấy chương trình này trên máy" if isinstance(error, FileNotFoundError) else "không có quyền chạy"
        return CommandResult(job_id, command, "failed", started, _now(), None, "", f"Không chạy được lệnh '{command[0]}': {reason}")
    try:
        stdout, stderr = process.communicate(timeout=timeout); timed_out = False
    except subprocess.TimeoutExpired:
        try: os.killpg(process.pid, signal.SIGKILL)  # dừng cả tiến trình con của lệnh
        except ProcessLookupError: pass
        stdout, stderr = process.communicate(); timed_out = True
        stderr = f"{stderr or ''}\nLệnh chạy quá {timeout:g} giây nên đã bị dừng".strip()
    status = "completed" if not timed_out and process.returncode == 0 else "failed"
    exit_code = None if timed_out else process.returncode
    return CommandResult(job_id, command, status, started, _now(), exit_code, (stdout or "")[-max_output:], (stderr or "")[-max_output:], timed_out)
