from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass


@dataclass(frozen=True)
class SandboxResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool


class PythonSandbox:
    """Chạy code trong tiến trình tạm; với code không tin cậy hãy cách ly bằng container/hệ điều hành."""

    def __init__(self, memory_mb: int | None = 512, cpu_seconds: int | None = 10):
        self.memory_mb, self.cpu_seconds = memory_mb, cpu_seconds

    def _limit_resources(self) -> None:
        # Chỉ chạy trên POSIX, trong tiến trình con trước khi thực thi code.
        import resource
        if self.memory_mb: resource.setrlimit(resource.RLIMIT_AS, (self.memory_mb * 1024 * 1024,) * 2)
        if self.cpu_seconds: resource.setrlimit(resource.RLIMIT_CPU, (self.cpu_seconds,) * 2)

    def run(self, code: str, timeout_seconds: float = 5) -> SandboxResult:
        # Môi trường tối giản: không truyền biến môi trường của máy (tránh lộ HF_TOKEN và khóa khác).
        environment = {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8"}
        with tempfile.TemporaryDirectory(prefix="local-ai-") as directory:
            try:
                completed = subprocess.run(
                    [sys.executable, "-I", "-c", code], cwd=directory, text=True, env=environment,
                    capture_output=True, timeout=timeout_seconds, check=False,
                    preexec_fn=self._limit_resources if os.name == "posix" else None,
                )
                return SandboxResult(completed.stdout, completed.stderr, completed.returncode, False)
            except subprocess.TimeoutExpired as error:
                return SandboxResult(error.stdout or "", error.stderr or "", -1, True)
