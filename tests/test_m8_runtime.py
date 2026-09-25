"""Mốc M8: runtime gộp từ repo Agent (local_ai/runtime).

Repo Agent gốc (commit 78a3e25) không có test; các test dưới đây kiểm tra đúng hành vi của `agent_runtime.py`
và `gateway.py` gốc, cộng các chỗ sửa để an toàn ghi trong docs/GOP_AGENT.md.
"""
import ast
import os
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from datetime import datetime
from pathlib import Path

from local_ai.runtime.auth import AuthError, require_token
from local_ai.runtime.executor import CommandResult, run_command
from local_ai.runtime.jobs import JobError, JobQueue

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


class ExecutorTests(unittest.TestCase):
    """Tương ứng `POST /v1/execute` của agent_runtime.py."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(); self.workspace = Path(self.directory.name) / "workspace"

    def tearDown(self):
        self.directory.cleanup()

    def test_success_records_job_fields(self):
        result = run_command([PYTHON, "-c", "print('xin chào')"], self.workspace)
        self.assertIsInstance(result, CommandResult)
        self.assertEqual((result.status, result.exit_code, result.stdout, result.stderr), ("completed", 0, "xin chào\n", ""))
        uuid.UUID(result.job_id)
        self.assertLessEqual(datetime.fromisoformat(result.started_at), datetime.fromisoformat(result.finished_at))
        self.assertEqual(result.argv[0], PYTHON)

    def test_runs_inside_workspace_and_creates_it(self):
        result = run_command([PYTHON, "-c", "import os; print(os.getcwd())"], self.workspace)
        self.assertTrue(self.workspace.is_dir())
        self.assertEqual(Path(result.stdout.strip()), self.workspace.resolve())

    def test_nonzero_exit_is_failed(self):
        result = run_command([PYTHON, "-c", "import sys; print('lỗi', file=sys.stderr); sys.exit(3)"], self.workspace)
        self.assertEqual((result.status, result.exit_code), ("failed", 3))
        self.assertIn("lỗi", result.stderr)

    def test_timeout_stops_the_command(self):
        start = time.monotonic()
        result = run_command([PYTHON, "-c", "import time; time.sleep(30)"], self.workspace, timeout=0.5)
        self.assertLess(time.monotonic() - start, 10)
        self.assertEqual((result.status, result.exit_code, result.timed_out), ("failed", None, True))
        self.assertIn("quá 0.5 giây", result.stderr)

    def test_output_keeps_last_characters(self):
        result = run_command([PYTHON, "-c", "print('a' * 100 + 'CUOI', end='')"], self.workspace, max_output=10)
        self.assertEqual(result.stdout, "aaaaaaCUOI")

    def test_missing_program_is_reported_not_raised(self):
        result = run_command(["lenh-khong-ton-tai-123"], self.workspace)
        self.assertEqual((result.status, result.exit_code), ("failed", None))
        self.assertIn("không tìm thấy chương trình", result.stderr)

    def test_no_shell_so_injection_is_plain_text(self):
        """Lỗ hổng của bản gốc: shell=True làm `ls; touch x` chạy cả lệnh thứ hai. Ở đây chuỗi chỉ là tham số."""
        for payload in ("a; touch pwned", "a && touch pwned", "a | touch pwned", "$(touch pwned)", "`touch pwned`", "a\ntouch pwned"):
            with self.subTest(payload):
                result = run_command(["echo", payload], self.workspace)
                self.assertEqual(result.stdout, payload + "\n")
                self.assertFalse((self.workspace / "pwned").exists())

    def test_string_command_is_rejected(self):
        for bad in ("ls; rm -rf ~", ["ls", 1], [], b"ls"):
            with self.subTest(bad), self.assertRaisesRegex(TypeError, "list"):
                run_command(bad, self.workspace)
        with self.assertRaisesRegex(ValueError, "timeout"): run_command(["ls"], self.workspace, timeout=0)


class AuthTests(unittest.TestCase):
    """Tương ứng `require_token` (runtime) và `auth` (gateway) của bản gốc."""

    def test_token_rules(self):
        with self.assertRaises(AuthError) as missing: require_token(None, "Bearer x")
        self.assertEqual(missing.exception.status, 503)
        for header in (None, "", "Bearer sai", "bi-mat-123", "bearer bi-mat-123"):
            with self.subTest(header), self.assertRaises(AuthError) as wrong:
                require_token("bi-mat-123", header)
            self.assertEqual(wrong.exception.status, 401)
        require_token("bi-mat-123", "Bearer bi-mat-123")


class JobQueueTests(unittest.TestCase):
    """Tương ứng `/v1/jobs` của gateway.py."""

    def test_job_lifecycle(self):
        queue = JobQueue()
        job = queue.create("liệt kê file")
        self.assertEqual((job.status, queue.get(job.id).instruction, len(queue)), ("queued", "liệt kê file", 1))
        self.assertEqual(queue.claim(job.id).status, "running")
        self.assertEqual(queue.submit_result(job.id, {"ok": True, "stdout": "a"}).status, "completed")
        self.assertEqual(queue.get(job.id).as_dict()["result"], {"ok": True, "stdout": "a"})
        other = queue.create("việc hai"); queue.claim(other.id)
        self.assertEqual(queue.submit_result(other.id, {"ok": "true"}).status, "failed")  # chỉ True thật mới là thành công

    def test_errors_have_http_status(self):
        queue = JobQueue(); job = queue.create("x")
        for action, status in ((lambda: queue.get("khong-co"), 404), (lambda: queue.submit_result(job.id, {"ok": True}), 409)):
            with self.assertRaises(JobError) as caught: action()
            self.assertEqual(caught.exception.status, status)
        queue.claim(job.id)
        with self.assertRaises(JobError) as twice: queue.claim(job.id)
        self.assertEqual(twice.exception.status, 409)
        for bad in ("", "x" * 10_001, None):
            with self.subTest(bad), self.assertRaises(ValueError): queue.create(bad)
        with self.assertRaises(ValueError): queue.submit_result(job.id, ["không phải object"])

    def test_queue_is_bounded(self):
        queue = JobQueue(max_jobs=2)
        first = queue.create("một"); queue.create("hai")
        with self.assertRaises(JobError) as full: queue.create("ba")
        self.assertEqual(full.exception.status, 429)
        queue.claim(first.id); queue.submit_result(first.id, {"ok": True})
        queue.create("ba")  # bỏ job đã xong cũ nhất để có chỗ
        with self.assertRaises(JobError): queue.get(first.id)
        self.assertEqual(len(queue), 2)


class ImportTests(unittest.TestCase):
    def test_import_has_no_side_effects_or_web_framework(self):
        with tempfile.TemporaryDirectory() as directory:
            code = "import sys, local_ai.runtime.executor, local_ai.runtime.jobs, local_ai.runtime.auth; print('fastapi' in sys.modules, 'uvicorn' in sys.modules)"
            result = subprocess.run([PYTHON, "-c", code], cwd=directory, env={**os.environ, "PYTHONPATH": str(ROOT)}, capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.strip(), "False False")
            self.assertEqual(os.listdir(directory), [])  # bản gốc tạo ./workspace ngay khi import

    def test_original_code_is_archived_not_imported(self):
        archived = ROOT / "archive" / "agent-goc"
        for name in ("agent_runtime.py", "gateway.py", "requirements.txt", "README_goc.md", "README.md"):
            self.assertTrue((archived / name).is_file(), name)
        self.assertIn("78a3e25bb4099d52118006f54b4e88aad3313ea2", (ROOT / "docs" / "GOP_AGENT.md").read_text(encoding="utf-8"))
        for path in (ROOT / "local_ai").rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("agent-goc", source, str(path))
            calls = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)]
            shell = [call.lineno for call in calls for keyword in call.keywords if keyword.arg == "shell" and not (isinstance(keyword.value, ast.Constant) and keyword.value.value is False)]
            self.assertEqual(shell, [], f"{path}: không được chạy lệnh qua shell")


if __name__ == "__main__":
    unittest.main()
