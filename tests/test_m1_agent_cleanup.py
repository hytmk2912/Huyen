"""Mốc M1: agent không sập khi công cụ lỗi, tách JSON từ câu trả lời model, dọn repo."""
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from local_ai.agents.loop import AutonomousAgent, extract_json_object
from local_ai.contracts import ToolCall
from local_ai.models.router import ModelRouter, ScriptedModelAdapter
from local_ai.tools.core import ToolRegistry, calculator

ROOT = Path(__file__).resolve().parent.parent


def calculator_tools():
    tools = ToolRegistry()
    tools.register("calculator", calculator)
    return tools


class ToolErrorTests(unittest.TestCase):
    def test_division_by_zero_and_syntax_error_do_not_raise(self):
        tools = calculator_tools()
        for expression, error in (("1/0", "ZeroDivisionError"), ("2 +", "SyntaxError")):
            result = tools.execute(ToolCall("calculator", {"expression": expression}))
            self.assertFalse(result.success)
            self.assertIn(error, result.output)

    def test_agent_survives_tool_error_and_retries(self):
        model = ScriptedModelAdapter("m", [
            "Dùng máy tính.",
            '{"tool": "calculator", "arguments": {"expression": "1/0"}}',
            '{"complete": false, "correction": "Chia cho 0 không được, tính 6 * 7."}',
            '{"tool": "calculator", "arguments": {"expression": "6 * 7"}}',
            '{"complete": true, "answer": "42"}',
        ])
        result = AutonomousAgent(ModelRouter([model]), calculator_tools(), 3).run("Tính 6 * 7")
        self.assertTrue(result.completed)
        self.assertEqual(result.answer, "42")
        self.assertTrue(any("ZeroDivisionError" in item for item in result.trace))


class JsonExtractionTests(unittest.TestCase):
    CASES = {
        "json thuần": '{"tool": "a"}',
        "chữ thừa trước và sau": 'Đây là quyết định: {"tool": "a"} Xong nhé.',
        "khối ```json": 'Kết quả:\n```json\n{"tool": "a"}\n```\n',
        "khối ``` không ghi ngôn ngữ": '```\n{"tool": "a"}\n```',
        "sau <think>": '<think>Có thể {"tool": "sai"} nhưng không.</think>\n{"tool": "a"}',
    }

    def test_each_format(self):
        for name, text in self.CASES.items():
            with self.subTest(name):
                self.assertEqual(extract_json_object(text), {"tool": "a"})

    def test_braces_inside_strings_and_nested_objects(self):
        text = 'Trả lời: {"tool": "echo", "arguments": {"text": "dấu { và } trong chuỗi"}} hết'
        self.assertEqual(extract_json_object(text)["arguments"]["text"], "dấu { và } trong chuỗi")

    def test_no_json_or_only_json_inside_think(self):
        self.assertEqual(extract_json_object("không có JSON"), {})
        self.assertEqual(extract_json_object('<think>{"tool": "a"}</think> chưa quyết'), {})
        self.assertEqual(extract_json_object('<think>đang nghĩ {"tool": "a"}'), {})
        self.assertEqual(extract_json_object("[1, 2]"), {})

    def test_agent_completes_with_messy_model_output(self):
        model = ScriptedModelAdapter("m", [
            "<think>Lập kế hoạch.</think> Dùng máy tính.",
            '<think>Gọi công cụ nào? {"tool": "sai"}</think>\n```json\n{"tool": "calculator", "arguments": {"expression": "6 * 7"}}\n```',
            'Đánh giá: {"complete": true, "answer": "42"} — xong.',
        ])
        result = AutonomousAgent(ModelRouter([model]), calculator_tools(), 2).run("Tính 6 * 7")
        self.assertTrue(result.completed)
        self.assertEqual(result.answer, "42")


class CleanupTests(unittest.TestCase):
    def test_corpus_is_archived_and_not_imported(self):
        for name in ("corpus.py", "corpus_10t.json", "smoke_real.json", "test_corpus.py", "README.md"):
            self.assertTrue((ROOT / "archive" / "corpus-10t" / name).exists(), name)
        self.assertFalse((ROOT / "local_ai" / "data" / "corpus.py").exists())
        self.assertFalse(list((ROOT / "configs").rglob("corpus_10t.json")) + list((ROOT / "configs").rglob("smoke_real.json")))
        for path in list((ROOT / "local_ai").rglob("*.py")) + list((ROOT / "tests").rglob("*.py")):
            if path.name != Path(__file__).name:
                self.assertNotIn("data.corpus", path.read_text(encoding="utf-8"), str(path))

    def test_secret_scan_command_still_works(self):
        from local_ai.data.secrets import scan_secrets
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.py").write_text("HF_TOKEN = 'hf_" + "a" * 30 + "'\n", encoding="utf-8")
            (root / "ok.py").write_text("print(1)\n", encoding="utf-8")
            self.assertEqual(scan_secrets(root), [str(root / "config.py")])
        result = subprocess.run([sys.executable, "-m", "local_ai.data", "secret-scan"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout)["findings"], [])

    def test_every_config_key_is_read_by_code(self):
        source = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "local_ai").rglob("*.py"))

        def keys(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    yield key
                    yield from keys(item)
            elif isinstance(value, list):
                for item in value:
                    yield from keys(item)

        for path in sorted((ROOT / "configs").rglob("*.json")):
            for key in set(keys(json.loads(path.read_text(encoding="utf-8")))) - {"_comment"}:
                used = f'"{key}"' in source or f"'{key}'" in source or re.search(rf"\b{re.escape(key)}\s*[:=]", source)
                self.assertTrue(used, f"{path.name}: khóa '{key}' không được code đọc")

    def test_old_repo_link_is_gone(self):
        tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split()
        allowed = {"TASKS.md", "memory.md", Path(__file__).relative_to(ROOT).as_posix()}
        for name in tracked:
            path = ROOT / name
            if name in allowed or name.startswith("archive/") or not path.is_file():
                continue
            self.assertNotIn("huyenb2404-ops", path.read_text(encoding="utf-8", errors="ignore"), name)

    def test_readme_follows_finetune_direction(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for phrase in ("sft.jsonl", "LoRA", "TASKS.md", "fine-tune"):
            self.assertIn(phrase, readme)
        self.assertNotIn("10.000.000.000.000", readme)


if __name__ == "__main__":
    unittest.main()
