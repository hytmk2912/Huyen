"""Sửa 2 lỗi bảo mật (26/9, chủ repo yêu cầu, không phải mốc; chủ repo đã thử thật cả hai lỗi):

1. TerminalTool đọc được repo cha qua git: thư mục làm việc của agent (`.runs/agent_tasks/<nhiệm vụ>/workspace`) nằm trong repo,
   `git log` được allowlist cho phép và git tự tìm `.git` ở thư mục cha, nên `git log -p` in được lịch sử và nội dung cả repo.
   Nay lệnh chạy với `GIT_CEILING_DIRECTORIES` là thư mục cha của thư mục làm việc.
2. PythonSandbox để lộ HF_TOKEN: notebook đặt `os.environ["HF_TOKEN"]`, code do model viết chạy trong sandbox thừa hưởng toàn bộ
   biến môi trường. Nay sandbox và lệnh của TerminalTool chỉ nhận môi trường tối thiểu (PATH, ngôn ngữ/mã hóa, HOME tạm).

Test dùng token giả và repo git tạm, không cần mạng.
"""
import os
import shutil
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from local_ai.runtime.executor import minimal_env
from local_ai.runtime.terminal import CommandFailed, CommandRule, TerminalConfig, TerminalTool
from local_ai.tools.sandbox import PythonSandbox

FAKE_TOKEN = "hf_" + "GiaDeThu" * 4  # token giả, ghép lúc chạy để secret-scan không báo chính file test
SECRETS = {"HF_TOKEN": FAKE_TOKEN, "OPENAI_API_KEY": "khoa-gia-" + "123456789", "GIT_DIR": "/khong/co", "GIT_WORK_TREE": "/khong/co"}


def git(directory: Path, *args: str) -> str:
    env = {**minimal_env(directory), "GIT_AUTHOR_NAME": "Thu", "GIT_AUTHOR_EMAIL": "thu@example.com", "GIT_COMMITTER_NAME": "Thu", "GIT_COMMITTER_EMAIL": "thu@example.com"}
    return subprocess.run(["git", *args], cwd=directory, env=env, capture_output=True, text=True, check=True).stdout


def make_repo(directory: Path, message: str, content: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    git(directory, "init", "-q")
    (directory / "bi_mat.txt").write_text(content, encoding="utf-8")
    git(directory, "add", "bi_mat.txt"); git(directory, "commit", "-q", "-m", message)


def terminal(root: Path, workspace: Path, **extra_rules) -> TerminalTool:
    config = TerminalConfig.from_file()
    rules = {**config.allowed_commands, **extra_rules}
    return TerminalTool(replace(config, workspace=workspace, log_path=root / "log.jsonl", allowed_commands=rules), enabled=True)


class SandboxEnvironmentTests(unittest.TestCase):
    def test_sandbox_code_cannot_read_token_or_real_home(self):
        code = "import json, os; print(json.dumps(dict(os.environ)))"
        with mock.patch.dict(os.environ, SECRETS):
            result = PythonSandbox().run(code)
        self.assertEqual(result.returncode, 0, result.stderr)
        seen = __import__("json").loads(result.stdout)
        self.assertNotIn(FAKE_TOKEN, result.stdout)
        for name in SECRETS:
            with self.subTest(name): self.assertNotIn(name, seen)
        self.assertLessEqual(set(seen), {"PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "PYTHONIOENCODING", "PYTHONUTF8"})
        self.assertNotEqual(seen["HOME"], os.path.expanduser("~"))
        self.assertIn("PATH", seen)

    def test_sandbox_still_runs_normal_code(self):
        result = PythonSandbox().run("print(sum(range(10)))")
        self.assertEqual((result.stdout.strip(), result.returncode), ("45", 0))


@unittest.skipUnless(shutil.which("git"), "máy không có git")
class TerminalEnvironmentTests(unittest.TestCase):
    def test_terminal_command_cannot_read_token(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, SECRETS):
            root = Path(directory)
            tool = terminal(root, root / "workspace", printenv=CommandRule(max_args=1))  # printenv chỉ thêm cho test, không có trong allowlist thật
            printed = tool("printenv")
        self.assertNotIn(FAKE_TOKEN, printed)
        names = {line.split("=", 1)[0] for line in printed.splitlines() if "=" in line}
        for name in SECRETS:
            with self.subTest(name): self.assertNotIn(name, names)
        self.assertIn("GIT_CEILING_DIRECTORIES", names)
        self.assertNotIn(f"HOME={os.path.expanduser('~')}", printed)

    def test_git_cannot_see_parent_repository(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, SECRETS):
            root = Path(directory).resolve()
            make_repo(root / "repo", "commit bí mật của repo cha", "nội dung bí mật của repo cha")
            workspace = root / "repo" / ".runs" / "agent_tasks" / "doc-ghi-chu" / "workspace"
            workspace.mkdir(parents=True)
            parent_commit = git(root / "repo", "rev-parse", "HEAD").strip()
            tool = terminal(root, workspace)
            for command in ("git log", "git log -p", "git status"):
                with self.subTest(command):
                    try: output = tool(command)
                    except CommandFailed as error: output = str(error)
                    self.assertNotIn("commit bí mật của repo cha", output)
                    self.assertNotIn("nội dung bí mật", output)
                    self.assertNotIn(parent_commit, output)
                    self.assertNotIn(parent_commit[:7], output)

    def test_git_still_works_for_a_repository_inside_the_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            make_repo(root / "repo", "commit của repo cha", "cha")
            workspace = root / "repo" / "workspace"
            make_repo(workspace, "commit trong thư mục làm việc", "con")
            output = terminal(root, workspace)("git log")
        self.assertIn("commit trong thư mục làm việc", output)
        self.assertNotIn("commit của repo cha", output)


if __name__ == "__main__":
    unittest.main()
