"""Mốc M13: agent chạy model thật trên Colab (Ollama + qwen3:4b) qua adapter kiểu OpenAI, làm 5 nhiệm vụ mẫu
bằng calculator và TerminalTool, in trace và tỉ lệ thành công.

Không cần mạng, GPU hay Ollama: notebook chỉ được kiểm tra hợp lệ và chạy lệnh bằng `--dry-run`; 5 nhiệm vụ chạy thật
với công cụ thật, còn model là câu trả lời mẫu hoặc một server HTTP giả nói chuẩn OpenAI (chạy ở 127.0.0.1).
"""
import contextlib
import http.server
import importlib.util
import io
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

from local_ai.agents.loop import AutonomousAgent
from local_ai.agents.tasks import DEFAULT_WORKSPACE, AgentTask, answer_matches, load_tasks, main as tasks_main, run_tasks
from local_ai.config.settings import find_model_config
from local_ai.data.secrets import SECRET
from local_ai.models.router import ModelRouter, ScriptedModelAdapter
from local_ai.runtime.terminal import register_terminal
from local_ai.tools.core import ToolRegistry

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "agent_colab.ipynb"
PLATFORM = ROOT / "configs" / "models" / "platform.json"
STEPS = ["gioi-thieu", "buoc-1-gpu", "buoc-2-tai-code", "buoc-3-cai-ollama", "buoc-4-chay-ollama", "buoc-5-tai-model", "buoc-6-kiem-tra", "buoc-7-chay-agent", "ket-qua"]
VIETNAMESE = re.compile(r"[ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]", re.I)


def builder():
    spec = importlib.util.spec_from_file_location("build_notebooks", ROOT / "notebooks" / "build.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def code_cells() -> dict[str, str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"}


class FakeOpenAIServer:
    """Server giả nói chuẩn OpenAI (POST /v1/chat/completions): trả lần lượt các câu trả lời trong `replies`, ghi lại request."""

    def __init__(self, replies):
        self.replies, self.requests = list(replies), []

    def __enter__(self):
        fake = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append({"path": self.path, "body": body})
                content = fake.replies.pop(0) if fake.replies else "hết câu trả lời"
                payload = json.dumps({"choices": [{"message": {"role": "assistant", "content": content}}]}).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(payload))); self.end_headers(); self.wfile.write(payload)
            def log_message(self, *args): pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True); self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        return self

    def __exit__(self, *exc):
        self.server.shutdown(); self.server.server_close(); self.thread.join()


def run_cli(argv):
    with contextlib.redirect_stdout(io.StringIO()) as output: code = tasks_main(argv)
    return code, output.getvalue()


class NotebookTests(unittest.TestCase):
    def test_notebook_is_valid_and_matches_builder(self):
        if importlib.util.find_spec("nbformat") is None: self.skipTest("thiếu nbformat (python -m pip install nbformat)")
        import nbformat
        nbformat.validate(nbformat.read(str(NOTEBOOK), as_version=4))
        build = builder()
        self.assertEqual(NOTEBOOK.read_text(encoding="utf-8"), build.render(build.agent_colab), "file .ipynb khác notebooks/build.py: hãy chạy python notebooks/build.py")
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual([cell["id"] for cell in notebook["cells"]], STEPS)
        self.assertEqual((notebook["metadata"]["accelerator"], notebook["metadata"]["colab"]["gpuType"]), ("GPU", "T4"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                with self.subTest(cell["id"]): self.assertEqual((cell["outputs"], cell["execution_count"]), ([], None))

    def test_cells_are_commented_and_versions_pinned(self):
        cells = code_cells(); build = builder()
        for number, (cell_id, source) in enumerate(cells.items(), start=1):
            with self.subTest(cell_id):
                first = source.splitlines()[0]
                self.assertTrue(first.startswith(f"# Bước {number}:"), first); self.assertRegex(first, VIETNAMESE)
        self.assertRegex(build.OLLAMA_VERSION, r"^\d+\.\d+\.\d+$")
        self.assertIn(f'env={{"OLLAMA_VERSION": "{build.OLLAMA_VERSION}"}}', cells["buoc-3-cai-ollama"])  # ghim bản Ollama (từ M16 chạy qua run, không qua shell)
        self.assertIn("zstd", cells["buoc-3-cai-ollama"])  # bản cài .tar.zst của Ollama cần zstd
        self.assertIn(f"ollama pull {build.OLLAMA_MODEL}", cells["buoc-5-tai-model"])
        self.assertEqual(find_model_config(PLATFORM, "ollama-colab").source, build.OLLAMA_MODEL)
        text = NOTEBOOK.read_text(encoding="utf-8")
        self.assertNotIn("pip install", text)  # phần agent chỉ dùng thư viện chuẩn
        self.assertIsNone(SECRET.search(text)); self.assertNotIn("HF_TOKEN", text)

    def test_notebook_commands_run_with_dry_run(self):
        lines = [command for source in code_cells().values() for command in re.findall(r'run\(f?"(python -m local_ai[^"]*)"', source)]  # từ M16: run("python -m ...")
        self.assertEqual(len(lines), 2)
        for line in lines:
            with self.subTest(line):
                result = subprocess.run([sys.executable, *shlex.split(line)[1:], "--dry-run"], cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)}, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                plan = json.loads(result.stdout)
                self.assertEqual((plan["model"]["name"], plan["model"]["base_url"], plan["model"]["source"]), ("ollama-colab", "http://localhost:11434/v1", "qwen3:4b"))
                self.assertEqual(len(plan["tasks"]), 5)


class TaskTests(unittest.TestCase):
    def test_five_tasks_use_both_tools_and_are_solvable(self):
        tasks = load_tasks()
        self.assertEqual(len(tasks), 5)
        self.assertEqual({tool for task in tasks for tool in task.tools}, {"calculator", "terminal"})
        self.assertTrue(any(set(task.tools) == {"calculator", "terminal"} for task in tasks))
        files = {path.name: path.read_text(encoding="utf-8") for path in DEFAULT_WORKSPACE.iterdir()}
        for task in tasks:
            if task.tools == ("terminal",):  # đáp án chỉ lấy được bằng cách đọc thư mục làm việc, model không đoán được
                with self.subTest(task.id): self.assertTrue(all(item in files or any(item in text for text in files.values()) for item in task.expected))

    def test_scripted_run_uses_real_tools_and_prints_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            code, output = run_cli(["--scripted", "--workspace", directory, "--output", str(Path(directory) / "report.json")])
            report = json.loads((Path(directory) / "report.json").read_text(encoding="utf-8"))
            logs = [json.loads(line) for line in (Path(directory) / "doc-ghi-chu" / "terminal.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(code, 0)
        self.assertIn("Tỉ lệ thành công: 5/5 (100%)", output)
        for phrase in ("=== tinh-bieu-thuc: ĐẠT ===", "tool[calculator]: 549", "tool[terminal]: bao_cao.md", "Mã số đơn hàng: DH-4827", "Công cụ đã gọi: terminal, calculator"):
            with self.subTest(phrase): self.assertIn(phrase, output)
        self.assertEqual((report["passed"], report["total"], report["success_rate"]), (5, 5, 1.0))
        self.assertEqual(logs[0]["command"], ["cat", "ghi_chu.txt"])  # TerminalTool thật, có ghi log

    def test_task_fails_on_wrong_answer_or_missing_tool(self):
        cases = {
            "câu trả lời thiếu 4": AgentTask("sai", "Tính 2 + 2.", ("4",), ("calculator",), ('plan', '{"tool": "calculator", "arguments": {"expression": "2 + 3"}}', '{"complete": true, "answer": "5"}')),
            "không gọi công cụ bắt buộc: calculator": AgentTask("thieu-cong-cu", "Đọc du_lieu.csv rồi tính tổng.", ("87",), ("terminal", "calculator"), ('plan', '{"tool": "terminal", "arguments": {"command": "cat du_lieu.csv"}}', '{"complete": true, "answer": "87"}')),
            "agent không hoàn thành": AgentTask("json-sai", "Tính 2 + 2.", ("4",), ("calculator",), ("plan", "tôi không biết JSON", "vẫn không", "hết")),
        }
        with tempfile.TemporaryDirectory() as directory:
            report = run_tasks(list(cases.values()), lambda task: ScriptedModelAdapter("mau", list(task.reference)), directory, max_iterations=2)
        self.assertEqual(report["passed"], 0)
        self.assertEqual([item["reason"] for item in report["tasks"]], list(cases))

    def test_terminal_stays_inside_task_workspace(self):
        task = AgentTask("ra-ngoai", "Đọc /etc/passwd.", ("root",), ("terminal",), ("plan", '{"tool": "terminal", "arguments": {"command": "cat /etc/passwd"}}', '{"complete": false, "correction": "thử lại"}',
                                                                              '{"tool": "terminal", "arguments": {"command": "rm ghi_chu.txt"}}', '{"complete": true, "answer": "root"}'))
        with tempfile.TemporaryDirectory() as directory:
            item = run_tasks([task], lambda task: ScriptedModelAdapter("mau", list(task.reference)), directory, max_iterations=2)["tasks"][0]
            self.assertTrue((Path(directory) / "ra-ngoai" / "workspace" / "ghi_chu.txt").exists())
        self.assertFalse(item["success"]); self.assertEqual(item["tools_used"], [])
        self.assertTrue(any("ngoài thư mục làm việc" in step for step in item["trace"]))

    def test_answer_matching(self):
        self.assertTrue(answer_matches("Khách phải trả 337.500 đồng.", ("337500",)))
        self.assertTrue(answer_matches("337 500 đ", ("337500",)))
        self.assertTrue(answer_matches("mã số là dh-4827", ("DH-4827",)))
        self.assertFalse(answer_matches("Có 2 file: bao_cao.md, du_lieu.csv", ("bao_cao.md", "du_lieu.csv", "ghi_chu.txt")))


class OpenAIAdapterTests(unittest.TestCase):
    def test_five_tasks_through_openai_compatible_server(self):
        replies = [reply for task in load_tasks() for reply in task.reference]
        with FakeOpenAIServer(replies) as server, tempfile.TemporaryDirectory() as directory:
            models = Path(directory) / "models.json"
            models.write_text(json.dumps({"models": [{"name": "ollama-gia", "backend": "openai_compatible", "base_url": server.base_url, "source": "qwen3:4b", "kind": "text", "capabilities": ["chat", "reasoning", "tool_calling"]}]}), encoding="utf-8")
            code, output = run_cli(["--model", "ollama-gia", "--models", str(models), "--workspace", str(Path(directory) / "runs")])
        self.assertEqual(code, 0)
        self.assertIn("Tỉ lệ thành công: 5/5 (100%)", output)
        self.assertEqual(len(server.requests), len(replies))
        self.assertTrue(all(request["path"] == "/v1/chat/completions" and request["body"]["model"] == "qwen3:4b" for request in server.requests))
        first, decision = (request["body"]["messages"][-1]["content"] for request in server.requests[:2])
        for text in (first, decision):  # model thật cần biết có công cụ nào và gọi với tham số gì
            with self.subTest(text[:40]):
                for phrase in ("calculator", "terminal", "expression", "command"): self.assertIn(phrase, text)
        self.assertIn('"tool": "<tool name>"', json.loads(decision)["instruction"])

    def test_models_list_has_ollama_colab(self):
        config = find_model_config(PLATFORM, "ollama-colab")
        self.assertEqual((config.backend, config.base_url, config.source), ("openai_compatible", "http://localhost:11434/v1", "qwen3:4b"))
        self.assertIn("reasoning", {capability.value for capability in config.capabilities})
        self.assertGreaterEqual(config.timeout_s, 120)  # lần gọi đầu Ollama còn phải nạp model


class ToolDescriptionTests(unittest.TestCase):
    def test_registry_describes_tools_and_agent_sends_them(self):
        registry = ToolRegistry(); register_terminal(registry)
        registry.register("calculator", lambda expression: expression, "tính biểu thức")
        self.assertEqual(set(registry.describe()), {"terminal", "calculator"})
        self.assertIn('"command"', registry.describe()["terminal"])
        prompts = []

        class Recorder(ScriptedModelAdapter):
            def generate(self, messages):
                prompts.append(messages[-1].content); return super().generate(messages)

        model = Recorder("ghi-lai", ["plan", '{"tool": "calculator", "arguments": "{\\"expression\\": \\"1 + 1\\"}"}', '{"complete": true, "answer": "1 + 1"}'])
        result = AutonomousAgent(ModelRouter([model]), registry, 1).run("Tính 1 + 1")
        self.assertTrue(result.completed)  # arguments dạng chuỗi JSON vẫn được hiểu
        self.assertIn("Available tools", prompts[0]); self.assertEqual(json.loads(prompts[1])["tools"], registry.describe())


if __name__ == "__main__":
    unittest.main()
