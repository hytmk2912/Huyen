from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass

from local_ai.runtime.executor import minimal_env


@dataclass(frozen=True)
class SandboxResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool


class PythonSandbox:
    """Chạy code trong tiến trình tạm; với code không tin cậy hãy cách ly bằng container/hệ điều hành.

    Code chạy với môi trường tối thiểu (`minimal_env`: PATH, ngôn ngữ/mã hóa, HOME là thư mục tạm), nên không đọc được
    HF_TOKEN hay khóa nào của tiến trình cha.
    """

    def run(self, code: str, timeout_seconds: float = 5) -> SandboxResult:
        with tempfile.TemporaryDirectory(prefix="local-ai-") as directory:
            try:
                completed = subprocess.run(
                    [sys.executable, "-I", "-c", code], cwd=directory, text=True,
                    capture_output=True, timeout=timeout_seconds, check=False, env=minimal_env(directory),
                )
                return SandboxResult(completed.stdout, completed.stderr, completed.returncode, False)
            except subprocess.TimeoutExpired as error:
                return SandboxResult(error.stdout or "", error.stderr or "", -1, True)
