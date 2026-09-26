"""Mốc M3: adapter gọi server local kiểu OpenAI (Ollama, llama.cpp, vLLM).

Test dựng server HTTP giả bằng `http.server` trên 127.0.0.1, không cần Ollama hay llama.cpp thật, không ra Internet.
"""
import contextlib
import io
import json
import os
import socket
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from local_ai.agents.loop import AutonomousAgent
from local_ai.config.settings import find_model_config, load_model_configs
from local_ai.contracts import Message
from local_ai.demo import main as demo_main
from local_ai.demo import run_demo
from local_ai.models.adapters import HuggingFaceModelAdapter, ModelConfig
from local_ai.models.openai_compatible import ModelServerError, OpenAICompatibleAdapter, check_local_url
from local_ai.models.router import ModelRouter, create_adapter
from local_ai.tools.core import ToolRegistry, calculator

ROOT = Path(__file__).resolve().parent.parent
PLATFORM = ROOT / "configs" / "models" / "platform.json"
AGENT_REPLIES = [
    "Use the calculator.",
    '{"tool": "calculator", "arguments": {"expression": "6 * 7"}}',
    '{"complete": true, "answer": "42"}',
]


class _QuietServer(ThreadingHTTPServer):
    daemon_threads = True
    def handle_error(self, request, client_address): pass


class FakeServer:
    """Server kiểu OpenAI giả: trả lần lượt các câu trong `replies`, hoặc mã lỗi, hoặc trả chậm."""

    def __init__(self, replies=(), status=200, delay=0.0, raw=None, headers=None):
        self.replies, self.status, self.delay, self.raw, self.headers = list(replies), status, delay, raw, headers or {}
        self.requests, self.release = [], threading.Event()

    def __enter__(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append({"path": self.path, "body": body, "headers": dict(self.headers)})
                if fake.delay: fake.release.wait(fake.delay)
                if fake.raw is not None: payload = fake.raw
                elif fake.status != 200: payload = b'{"error": "boom"}'
                else: payload = json.dumps({"choices": [{"message": {"role": "assistant", "content": fake.replies.pop(0) if fake.replies else "ok"}}]}).encode()
                try:
                    self.send_response(fake.status)
                    for key, value in {"Content-Type": "application/json", "Content-Length": str(len(payload)), **fake.headers}.items(): self.send_header(key, value)
                    self.end_headers(); self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError): pass
            def log_message(self, *args): pass

        self.server = _QuietServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True); self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        return self

    def __exit__(self, *exc):
        self.release.set(); self.server.shutdown(); self.server.server_close(); self.thread.join()


def server_config(base_url, **options):
    return ModelConfig.from_dict({"name": "local-test", "backend": "openai_compatible", "base_url": base_url, "source": "qwen-test", "capabilities": ["chat", "reasoning"], **options})


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0)); return sock.getsockname()[1]


def calculator_tools():
    tools = ToolRegistry(); tools.register("calculator", calculator); return tools


class RequestTests(unittest.TestCase):
    def test_success_sends_openai_payload_and_returns_content(self):
        with FakeServer(["chào bạn"]) as server:
            answer = OpenAICompatibleAdapter(server_config(server.base_url)).generate([Message("system", "Trả lời ngắn."), Message("user", "xin chào")])
        self.assertEqual(answer, "chào bạn")
        request = server.requests[0]
        self.assertEqual(request["path"], "/v1/chat/completions")
        self.assertEqual(request["body"], {"model": "qwen-test", "messages": [{"role": "system", "content": "Trả lời ngắn."}, {"role": "user", "content": "xin chào"}], "stream": False,
                                           "temperature": 0.0, "seed": 0})  # M19: temperature 0 và seed cố định để chấm lặp lại được
        self.assertNotIn("Authorization", request["headers"])

    def test_api_key_is_read_only_from_environment(self):
        with FakeServer(["ok"]) as server, mock.patch.dict(os.environ, {"TEST_LOCAL_LLM_KEY": "khoa-thu"}):
            OpenAICompatibleAdapter(server_config(server.base_url, api_key_env="TEST_LOCAL_LLM_KEY")).generate([Message("user", "hi")])
        self.assertEqual(server.requests[0]["headers"]["Authorization"], "Bearer khoa-thu")
        for model in json.loads(PLATFORM.read_text(encoding="utf-8"))["models"]:
            self.assertFalse({"api_key", "key", "token"} & set(model), model["name"])

    def test_images_are_rejected_clearly(self):
        with self.assertRaisesRegex(ValueError, "chưa hỗ trợ gửi ảnh"):
            OpenAICompatibleAdapter(server_config("http://127.0.0.1:1/v1")).generate([Message("user", "xem", images=("anh.png",))])


class ErrorTests(unittest.TestCase):
    def assert_server_error(self, config, *phrases):
        with self.assertRaises(ModelServerError) as caught:
            OpenAICompatibleAdapter(config).generate([Message("user", "hi")])
        for phrase in phrases: self.assertIn(phrase, str(caught.exception))

    def test_http_500(self):
        with FakeServer(status=500) as server: self.assert_server_error(server_config(server.base_url), "HTTP 500", "boom")

    def test_slow_server_times_out(self):
        with FakeServer(delay=10) as server:
            start = time.monotonic()
            self.assert_server_error(server_config(server.base_url, timeout_s=0.3), "không trả lời trong 0.3 giây")
            self.assertLess(time.monotonic() - start, 5)

    def test_server_not_running(self):
        self.assert_server_error(server_config(f"http://127.0.0.1:{free_port()}/v1"), "Không kết nối được", "Server đã chạy chưa")

    def test_bad_response_and_redirect(self):
        with FakeServer(raw=b"not json") as server: self.assert_server_error(server_config(server.base_url), "không đúng chuẩn OpenAI")
        with FakeServer(status=302, headers={"Location": "http://8.8.8.8/v1/chat/completions"}) as server:
            self.assert_server_error(server_config(server.base_url), "HTTP 302")
            self.assertEqual(len(server.requests), 1)


class LocalAddressTests(unittest.TestCase):
    def test_local_and_private_addresses_are_allowed(self):
        for url in ("http://127.0.0.1:8080/v1", "http://localhost:11434/v1", "http://[::1]:8000/v1", "http://192.168.1.5:8000/v1", "http://10.0.0.2/v1", "http://100.100.1.1/v1"):
            with self.subTest(url): check_local_url(url)
        check_local_url("http://gpu-box.lan:8000/v1", resolve=lambda host, port: [(None, None, None, "", ("192.168.1.20", port))])

    def test_internet_addresses_are_rejected_before_sending(self):
        with self.assertRaisesRegex(ValueError, "Internet"): check_local_url("http://8.8.8.8/v1")
        with self.assertRaisesRegex(ValueError, "Internet"): check_local_url("https://api.openai.com/v1", resolve=lambda host, port: [(None, None, None, "", ("104.18.6.192", port))])
        adapter = OpenAICompatibleAdapter(server_config("http://8.8.8.8/v1"))
        with mock.patch.object(adapter._opener, "open") as opened, self.assertRaisesRegex(ValueError, "Internet"):
            adapter.generate([Message("user", "hi")])
        opened.assert_not_called()

    def test_config_is_validated(self):
        for options, message in (({"base_url": None}, "base_url"), ({"base_url": "ftp://127.0.0.1/v1"}, "base_url"), ({"backend": "openai"}, "backend"), ({"timeout_s": 0}, "timeout_s")):
            with self.subTest(options), self.assertRaisesRegex(ValueError, message):
                server_config(**{"base_url": "http://127.0.0.1:8080/v1", **options})


class ConfigRoutingTests(unittest.TestCase):
    def test_platform_declares_ollama_and_llamacpp(self):
        ollama, llamacpp = find_model_config(PLATFORM, "ollama"), find_model_config(PLATFORM, "llamacpp")
        self.assertEqual((ollama.backend, ollama.base_url), ("openai_compatible", "http://localhost:11434/v1"))
        self.assertEqual((llamacpp.backend, llamacpp.base_url), ("openai_compatible", "http://localhost:8080/v1"))
        adapters = {config.name: create_adapter(config) for config in load_model_configs(PLATFORM)}
        self.assertIsInstance(adapters["ollama"], OpenAICompatibleAdapter)
        self.assertIsInstance(adapters["primary"], HuggingFaceModelAdapter)
        self.assertIsInstance(ModelRouter.from_configs([ollama]).select("reasoning"), OpenAICompatibleAdapter)

    def test_agent_runs_through_router_built_from_config(self):
        with FakeServer(AGENT_REPLIES) as server:
            result = AutonomousAgent(ModelRouter.from_configs([server_config(server.base_url)]), calculator_tools(), 2).run("Tính 6 * 7")
        self.assertTrue(result.completed)
        self.assertEqual(result.answer, "42")
        self.assertEqual(len(server.requests), 3)

    def test_agent_does_not_crash_when_server_fails(self):
        with FakeServer(status=500) as server:
            result = AutonomousAgent(ModelRouter.from_configs([server_config(server.base_url)]), calculator_tools(), 2).run("Tính 6 * 7")
        self.assertFalse(result.completed)
        self.assertIn("HTTP 500", result.answer)
        self.assertTrue(any(item.startswith("model error:") for item in result.trace))
        result = AutonomousAgent(ModelRouter.from_configs([server_config(f"http://127.0.0.1:{free_port()}/v1")]), calculator_tools(), 2).run("x")
        self.assertIn("Không kết nối được", result.answer)

    def test_finetune_rejects_server_models(self):
        from local_ai.training.finetune import describe, load_finetune_config
        with self.assertRaisesRegex(ValueError, "server"): describe(load_finetune_config(ROOT / "configs" / "training" / "sft.json", base_model="ollama"))


class DemoTests(unittest.TestCase):
    def models_file(self, directory, base_url):
        path = Path(directory) / "models.json"
        path.write_text(json.dumps({"models": [{"name": "local-test", "backend": "openai_compatible", "base_url": base_url, "source": "qwen-test", "capabilities": ["chat"]}]}), encoding="utf-8")
        return path

    def test_demo_runs_through_adapter(self):
        replies = ["Calculate.", '{"tool": "calculator", "arguments": {"expression": "21 * 2"}}', '{"complete": true, "answer": "The result is 42."}']
        with FakeServer(replies) as server, tempfile.TemporaryDirectory() as directory:
            self.assertEqual(run_demo("local-test", self.models_file(directory, server.base_url)), "The result is 42.")
        self.assertEqual(server.requests[0]["body"]["model"], "qwen-test")

    def test_demo_command_reports_server_down(self):
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as caught:
            demo_main(["--model", "local-test", "--models", str(self.models_file(directory, f"http://127.0.0.1:{free_port()}/v1"))])
        self.assertEqual(caught.exception.code, 1)
        self.assertIn("Server đã chạy chưa", stderr.getvalue())


class ReadmeTests(unittest.TestCase):
    def test_readme_explains_ollama_llamacpp_and_vllm(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for phrase in ("ollama serve", "http://localhost:11434/v1", "llama-server", "http://localhost:8080/v1", "vllm serve", "api_key_env", "python -m local_ai.demo --model ollama"):
            self.assertIn(phrase, readme)
        entry = json.loads(readme.split('```json\n', 1)[1].split("\n```", 1)[0])
        self.assertIsInstance(create_adapter(ModelConfig.from_dict(entry)), OpenAICompatibleAdapter)


if __name__ == "__main__":
    unittest.main()
