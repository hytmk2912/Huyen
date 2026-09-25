"""Mốc M9: TerminalTool an toàn (allowlist, không shell, timeout, log) và gateway HTTP tắt mặc định."""
import contextlib
import io
import json
import os
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path

from local_ai.agents.loop import AutonomousAgent
from local_ai.models.router import ModelRouter, ScriptedModelAdapter
from local_ai.runtime.gateway import DEFAULT_CONFIG, GatewayConfig, GatewayRefused, create_server, main as gateway_main
from local_ai.runtime.terminal import CommandFailed, CommandRejected, CommandRule, TerminalConfig, TerminalTool, register_terminal
from local_ai.tools.core import ToolRegistry

ORIGINAL_ALLOWLIST = {"pwd", "ls", "find", "cat", "head", "tail", "grep", "git", "python", "python3", "node", "npm", "uname", "whoami", "date"}  # ALLOWED_PREFIXES của repo Agent gốc


class TerminalCase(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(); root = Path(self.directory.name)
        self.workspace, self.log = root / "workspace", root / "log.jsonl"
        self.workspace.mkdir(); (self.workspace / "ghi_chu.txt").write_text("dòng một\ndòng hai\n", encoding="utf-8")
        self.config = replace(TerminalConfig.from_file(), workspace=self.workspace, log_path=self.log)
        self.tool = TerminalTool(self.config, enabled=True)

    def tearDown(self):
        self.directory.cleanup()

    def log_entries(self):
        return [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines()]


class ConfigTests(unittest.TestCase):
    def test_shipped_config_is_off_and_matches_original_allowlist(self):
        config = TerminalConfig.from_file()
        self.assertFalse(config.enabled)
        self.assertEqual(set(config.allowed_commands), ORIGINAL_ALLOWLIST)
        self.assertEqual(config.allowed_commands["git"].subcommands, ("status", "log"))
        self.assertEqual(config.allowed_commands["python"].exact_args, (("--version",),))
        with self.assertRaisesRegex(CommandRejected, "đang tắt"): TerminalTool(config)("ls")


class InjectionTests(TerminalCase):
    PAYLOADS = {"dấu ;": "ls; touch pwned", "dấu &&": "ls && touch pwned", "dấu |": "ls | touch pwned", "$()": "cat $(touch pwned)",
                "backtick": "cat `touch pwned`", "xuống dòng": "ls\ntouch pwned", "\\r": "ls\rtouch pwned", "dấu >": "ls > pwned", "dấu <": "cat < ghi_chu.txt", "&": "ls & touch pwned"}

    def test_shell_metacharacters_are_rejected_before_running(self):
        for label, payload in self.PAYLOADS.items():
            with self.subTest(label):
                with self.assertRaisesRegex(CommandRejected, "ký tự điều khiển shell"): self.tool(payload)
                with self.assertRaisesRegex(CommandRejected, "ký tự điều khiển shell"): self.tool(["cat", payload])  # dạng list cũng bị chặn
        self.assertFalse((self.workspace / "pwned").exists())
        self.assertTrue(all(entry["allowed"] is False for entry in self.log_entries()))

    def test_commands_outside_allowlist_are_rejected(self):
        for command in ("rm -rf ghi_chu.txt", "curl http://example.com", "bash", "sh", "python -c print(1)", "/bin/ls", "./ls", "git push", "git -c core.pager=id log", "npm install x", "sudo ls"):
            with self.subTest(command), self.assertRaises(CommandRejected): self.tool(command)
        self.assertTrue((self.workspace / "ghi_chu.txt").exists())

    def test_dangerous_options_and_paths_are_rejected(self):
        for command in ("find . -delete", "find . -execdir id", "find -L .", "git log --output=x.txt", "git log -ox.txt", "tail -f ghi_chu.txt", "tail -fn1 ghi_chu.txt", "grep -R dòng .",
                        "cat /etc/passwd", "cat ../log.jsonl", "head -n1 ../../etc/hostname", "grep --file=/etc/passwd x", "date -s2020-01-01", "date --set=2020-01-01", "date -us2020-01-01", "tail -qf ghi_chu.txt", "find . -exec+", "grep -rR dòng .", "grep -f/etc/passwd x", "grep -f../log.jsonl x", "ls " + "a " * 9):
            with self.subTest(command), self.assertRaises(CommandRejected): self.tool(command)


class AllowedCommandTests(TerminalCase):
    def test_allowed_commands_run_in_workspace_and_are_logged(self):
        self.assertEqual(self.tool("ls"), "ghi_chu.txt\n")
        self.assertEqual(self.tool(["cat", "ghi_chu.txt"]), "dòng một\ndòng hai\n")
        self.assertEqual(self.tool("head -n 1 ghi_chu.txt"), "dòng một\n")
        self.assertEqual(self.tool("grep hai ghi_chu.txt"), "dòng hai\n")
        self.assertEqual(Path(self.tool("pwd").strip()), self.workspace.resolve())
        self.assertEqual(self.tool.check("git status"), ["git", "status"])
        entries = self.log_entries()
        self.assertEqual(len(entries), 5)
        self.assertTrue(all(entry["allowed"] and entry["exit_code"] == 0 and entry["job_id"] for entry in entries))
        self.assertEqual(entries[1]["command"], ["cat", "ghi_chu.txt"])

    def test_failing_command_raises_with_exit_code(self):
        with self.assertRaisesRegex(CommandFailed, "mã thoát"): self.tool("cat khong_co.txt")
        self.assertEqual(self.log_entries()[-1]["status"], "failed")

    def test_timeout_stops_the_command(self):
        tool = TerminalTool(replace(self.config, timeout_s=0.5, allowed_commands={"sleep": CommandRule(max_args=1)}), enabled=True)
        start = time.monotonic()
        with self.assertRaisesRegex(CommandFailed, "quá 0.5 giây"): tool("sleep 30")
        self.assertLess(time.monotonic() - start, 10)
        self.assertTrue(self.log_entries()[-1]["timed_out"])


class AgentRegistrationTests(TerminalCase):
    def run_agent(self, tool):
        registry = ToolRegistry(); register_terminal(registry, tool)
        model = ScriptedModelAdapter("m", ["Đọc file ghi chú.", '{"tool": "terminal", "arguments": {"command": "cat ghi_chu.txt"}}', '{"complete": true, "answer": "Có 2 dòng."}'])
        return AutonomousAgent(ModelRouter([model]), registry, 1).run("File ghi chú có mấy dòng?")

    def test_agent_uses_enabled_terminal(self):
        result = self.run_agent(self.tool)
        self.assertTrue(result.completed)
        self.assertTrue(any("tool[terminal]: dòng một" in item for item in result.trace))

    def test_disabled_terminal_fails_safely(self):
        result = self.run_agent(TerminalTool(self.config))
        self.assertFalse(result.completed)
        self.assertTrue(any("đang tắt" in item for item in result.trace))


class GatewayTests(unittest.TestCase):
    TOKEN = "token-thu-nghiem-dai-hon-16"

    def test_refuses_unsafe_configs(self):
        self.assertFalse(GatewayConfig.from_file(DEFAULT_CONFIG).enabled)
        environ = {"LOCAL_AI_GATEWAY_TOKEN": self.TOKEN}
        for config, message in ((GatewayConfig(), "đang tắt"), (GatewayConfig(enabled=True, host="0.0.0.0"), "127.0.0.1"), (GatewayConfig(enabled=True, host="localhost"), "127.0.0.1")):
            with self.subTest(message), self.assertRaisesRegex(GatewayRefused, message): create_server(config, environ)
        for environ in ({}, {"LOCAL_AI_GATEWAY_TOKEN": "ngan"}):
            with self.assertRaisesRegex(GatewayRefused, "token"): create_server(GatewayConfig(enabled=True, port=0), environ)
        with contextlib.redirect_stdout(io.StringIO()) as output: self.assertEqual(gateway_main([]), 1)
        self.assertIn("Không bật gateway", output.getvalue())

    def test_job_flow_requires_token(self):
        server = create_server(GatewayConfig(enabled=True, port=0), {"LOCAL_AI_GATEWAY_TOKEN": self.TOKEN})
        self.assertEqual(server.server_address[0], "127.0.0.1")
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True); thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        def call(method, path, body=None, token=self.TOKEN):
            headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})}
            request = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(), headers=headers, method=method)
            try:
                with opener.open(request, timeout=5) as response: return response.status, json.loads(response.read())
            except urllib.error.HTTPError as error: return error.code, json.loads(error.read())

        try:
            self.assertEqual(call("GET", "/health", token=None), (200, {"ok": True}))
            self.assertEqual(call("POST", "/v1/jobs", {"instruction": "ls"}, token=None)[0], 401)
            self.assertEqual(call("POST", "/v1/jobs", {"instruction": "ls"}, token="sai-token-sai-token")[0], 401)
            status, created = call("POST", "/v1/jobs", {"instruction": "liệt kê file"})
            self.assertEqual((status, created["status"]), (201, "queued"))
            job = f"/v1/jobs/{created['job_id']}"
            self.assertEqual(call("GET", job)[1]["instruction"], "liệt kê file")
            self.assertEqual(call("POST", job + "/claim")[1]["status"], "running")
            self.assertEqual(call("POST", job + "/claim")[0], 409)
            self.assertEqual(call("POST", job + "/result", {"ok": True, "stdout": "a"})[1]["status"], "completed")
            self.assertEqual(call("GET", "/v1/jobs/00000000-0000-0000-0000-000000000000")[0], 404)
            self.assertEqual(call("POST", "/v1/jobs", {"instruction": ""})[0], 400)
        finally:
            server.shutdown(); server.server_close(); thread.join()


if __name__ == "__main__":
    unittest.main()
