import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_ai.agents.loop import AutonomousAgent
from local_ai.contracts import ToolCall
from local_ai.evaluation.suites import EvalCase, load_cases, parse_number, run_suites, score
from local_ai.models.router import ModelRouter, ScriptedModelAdapter
from local_ai.tools.core import ToolRegistry, calculator
from local_ai.tools.files import register_file_tools
from local_ai.tools.sandbox import PythonSandbox


class FlakyModel(ScriptedModelAdapter):
    def __init__(self, responses):
        super().__init__("flaky", responses); self.failed = False
    def generate(self, messages):
        if not self.failed: self.failed = True; raise ConnectionError("model server restarted")
        return super().generate(messages)


class AgentToolsAndRecoveryTests(unittest.TestCase):
    def test_file_tools_stay_inside_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "note.txt").write_text("xin chào", encoding="utf-8")
            registry = ToolRegistry(); register_file_tools(registry, directory)
            self.assertEqual(registry.execute(ToolCall("read_file", {"path": "note.txt"})).output, "xin chào")
            self.assertIn("note.txt", registry.execute(ToolCall("list_files", {})).output)
            escape = registry.execute(ToolCall("read_file", {"path": "../../etc/passwd"}))
            self.assertFalse(escape.success); self.assertIn("outside", escape.output)
            self.assertFalse(registry.execute(ToolCall("write_file", {"path": "x", "content": "y"})).success)
            writable = ToolRegistry(); register_file_tools(writable, directory, allow_write=True)
            self.assertTrue(writable.execute(ToolCall("write_file", {"path": "sub/out.txt", "content": "ok"})).success)
            self.assertEqual((Path(directory) / "sub" / "out.txt").read_text(), "ok")

    def test_unexpected_tool_error_is_recovered(self):
        registry = ToolRegistry()
        def broken(): raise RuntimeError("disk full")
        registry.register("broken", broken)
        result = registry.execute(ToolCall("broken"))
        self.assertFalse(result.success); self.assertIn("disk full", result.output)

    def test_agent_recovers_from_model_failure(self):
        model = FlakyModel(['{"tool": "calculator", "arguments": {"expression": "6 * 7"}}', '{"complete": true, "answer": "42"}'])
        tools = ToolRegistry(); tools.register("calculator", calculator)
        result = AutonomousAgent(ModelRouter([model]), tools, 2).run("Calculate 6 * 7")
        self.assertTrue(result.completed); self.assertEqual(result.answer, "42")
        self.assertTrue(any("model failure" in item for item in result.trace))

    @unittest.skipUnless(os.name == "posix", "giới hạn tài nguyên chỉ có trên POSIX")
    def test_sandbox_hides_secrets_and_limits_memory(self):
        with mock.patch.dict(os.environ, {"HF_TOKEN": "secret-value"}):
            result = PythonSandbox().run("import os; print(os.environ.get('HF_TOKEN'))")
        self.assertEqual(result.stdout.strip(), "None")
        result = PythonSandbox(memory_mb=256).run("x = bytearray(1024 * 1024 * 1024)")
        self.assertNotEqual(result.returncode, 0)


class SuiteEvaluationTests(unittest.TestCase):
    def test_bundled_cases_cover_vietnamese_and_trading(self):
        cases = load_cases("data/eval/vi_trading_eval.jsonl")
        self.assertEqual({case.suite for case in cases}, {"vietnamese", "trading"})
        answers = {case.prompt: case.expected for case in cases}
        report = run_suites(cases, lambda prompt: answers[prompt])
        self.assertEqual(report["accuracy"], 1.0)
        self.assertEqual(report["suites"]["trading"]["cases"], 4)

    def test_scoring_rules(self):
        self.assertTrue(score(EvalCase("a", "vietnamese", "p", "Hà Nội", "contains"), "Đáp án: hà nội."))
        self.assertTrue(score(EvalCase("b", "trading", "p", "1000000", "numeric"), "1.000.000 đồng"))
        self.assertTrue(score(EvalCase("c", "trading", "p", "12.5", "numeric", 0.01), "Khoảng 12,5%"))
        self.assertFalse(score(EvalCase("d", "trading", "p", "3", "numeric"), "không biết"))
        self.assertTrue(score(EvalCase("e", "vietnamese", "p", "lạnh", "exact"), " Lạnh. "))
        self.assertEqual(parse_number("0.125"), 0.125)

    def test_report_groups_failures_and_survives_runner_errors(self):
        cases = [EvalCase("v", "vietnamese", "p1", "a"), EvalCase("t", "trading", "p2", "5", "numeric")]
        def runner(prompt):
            if prompt == "p2": raise TimeoutError
            return "a"
        report = run_suites(cases, runner)
        self.assertEqual(report["failures"], ["t"])
        self.assertEqual(report["suites"]["vietnamese"]["accuracy"], 1.0)
        self.assertEqual(report["suites"]["trading"]["accuracy"], 0.0)


if __name__ == "__main__": unittest.main()
