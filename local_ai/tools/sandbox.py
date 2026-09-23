from __future__ import annotations

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

    def run(self, code: str, timeout_seconds: float = 5) -> SandboxResult:
        with tempfile.TemporaryDirectory(prefix="local-ai-") as directory:
            try:
                completed = subprocess.run(
                    [sys.executable, "-I", "-c", code], cwd=directory, text=True,
                    capture_output=True, timeout=timeout_seconds, check=False,
                )
                return SandboxResult(completed.stdout, completed.stderr, completed.returncode, False)
            except subprocess.TimeoutExpired as error:
                return SandboxResult(error.stdout or "", error.stderr or "", -1, True)
