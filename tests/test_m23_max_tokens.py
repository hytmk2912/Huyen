"""Mốc M23: model qua server (Ollama, llama.cpp, vLLM) nhận `max_tokens` = `max_new_tokens`.

Trước M23 adapter kiểu OpenAI không gửi `max_tokens`, nên `max_new_tokens` (trong danh sách model hay `--max-new-tokens` của
lệnh chấm) không giới hạn được độ dài câu trả lời của model qua server (phát hiện khi làm M18). Test dùng server HTTP giả
trong máy, kiểm tra payload server nhận được; không cần mạng hay Ollama thật.
"""
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

from local_ai.config.settings import find_model_config
from local_ai.contracts import Message
from local_ai.evaluation.__main__ import main as evaluation_main
from local_ai.models.openai_compatible import OpenAICompatibleAdapter

ROOT = Path(__file__).resolve().parent.parent
PLATFORM = ROOT / "configs" / "models" / "platform.json"


def m3_helpers():
    spec = importlib.util.spec_from_file_location("m3_helpers_m23", Path(__file__).with_name("test_m3_local_server.py"))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


M3 = m3_helpers()


def payload(**options) -> dict:
    """Payload mà server giả nhận được khi adapter hỏi một câu."""
    with M3.FakeServer(["chào"]) as server:
        OpenAICompatibleAdapter(M3.server_config(server.base_url, **options)).generate([Message("user", "xin chào")])
    return server.requests[0]["body"]


class MaxTokensTests(unittest.TestCase):
    def test_default_max_new_tokens_is_sent_as_max_tokens(self):
        self.assertEqual(payload()["max_tokens"], 512)  # max_new_tokens mặc định của ModelConfig

    def test_configured_max_new_tokens_is_sent(self):
        for value in (1, 64, 300, 4096):
            with self.subTest(value): self.assertEqual(payload(max_new_tokens=value)["max_tokens"], value)

    def test_eval_command_override_reaches_the_server(self):
        case = {"id": "en-thu", "language": "en", "group": "reasoning", "prompt": "What is 2 + 2?", "scoring": "contains", "expected": "4", "reference": "4"}
        with M3.FakeServer(["4", "4"]) as server, tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cases.jsonl").write_text(json.dumps(case) + "\n", encoding="utf-8")
            (root / "models.json").write_text(json.dumps({"models": [{"name": "may-chu", "backend": "openai_compatible", "base_url": server.base_url, "source": "qwen-test",
                                                                      "capabilities": ["chat"], "max_new_tokens": 200}]}), encoding="utf-8")
            argv = ["--model", "may-chu", "--models", str(root / "models.json"), "--cases", str(root / "cases.jsonl")]
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(evaluation_main([*argv, "--output", str(root / "a")]), 0)  # giá trị trong danh sách model
                self.assertEqual(evaluation_main([*argv, "--output", str(root / "b"), "--max-new-tokens", "32"]), 0)  # giá trị tự đặt khi chấm
        self.assertEqual([request["body"]["max_tokens"] for request in server.requests], [200, 32])

    def test_platform_server_models_send_their_limit(self):
        for name in ("ollama", "ollama-colab", "llamacpp", "ollama-light-da-train"):
            with self.subTest(name):
                config = find_model_config(PLATFORM, name)
                with M3.FakeServer(["ok"]) as server:
                    OpenAICompatibleAdapter(M3.server_config(server.base_url, max_new_tokens=config.max_new_tokens)).generate([Message("user", "xin chào")])
                self.assertEqual(server.requests[0]["body"]["max_tokens"], config.max_new_tokens)


if __name__ == "__main__":
    unittest.main()
