import io
import json
import tempfile
import time
import unittest
from pathlib import Path

from local_ai.agents.loop import AgentBudget, AutonomousAgent
from local_ai.contracts import ToolCall
from local_ai.data.corpus import Source, build_one, fetch_hf_text
from local_ai.data.limits import StorageBudget, StorageLimitReached, Throttle, copy_limited
from local_ai.models.router import ModelRouter, ScriptedModelAdapter
from local_ai.tools.core import ToolRegistry, calculator

CORPUS = "configs/datasets/corpus_10t.json"
ROWS = [{"text": f"Dòng dữ liệu số {index} " + "nội dung " * 20} for index in range(6)]


def source():
    config = json.loads(Path(CORPUS).read_text())
    return Source.from_dict({**config["sources"][-2], "source_id": "vi-test", "options": {"text_field": "text"}})


class FakeClock:
    def __init__(self): self.now = 0.0
    def __call__(self): return self.now
    def sleep(self, seconds): self.now += seconds


class ResourceLimitTests(unittest.TestCase):
    def test_storage_budget_stops_and_download_resumes(self):
        loader = lambda name, subset=None, **kwargs: iter(ROWS)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "raw.txt"; row_bytes = len(ROWS[0]["text"].encode()) + 2
            budget = StorageBudget(Path(directory), 1); budget.limit = row_bytes * 3; budget.used = 0
            with self.assertRaises(StorageLimitReached): fetch_hf_text(source(), target, loader, budget)
            self.assertFalse(target.exists())
            self.assertEqual(target.with_suffix(".progress").read_text(), "3")  # dòng đầu không có dấu phân cách nên vừa 3 dòng
            fetch_hf_text(source(), target, loader)
            lines = target.read_text(encoding="utf-8").split("\n\n")
            self.assertEqual(len(lines), 6)
            self.assertEqual(lines[3], ROWS[3]["text"].strip())

    def test_build_pauses_instead_of_failing(self):
        config = json.loads(Path(CORPUS).read_text())
        loader = lambda name, subset=None, **kwargs: iter(ROWS)
        with tempfile.TemporaryDirectory() as directory:
            config.update({"storage_root": directory, "tokenizer": {"kind": "utf8_bytes", "name": "test"}, "upload": {"enabled": False}, "max_local_storage_gb": 1e-7})
            result = build_one(source(), config, loader=loader)
        self.assertEqual((result["status"], result["reason"]), ("paused", "storage_limit"))

    def test_throttle_keeps_average_rate(self):
        clock = FakeClock(); throttle = Throttle(8, clock=clock, sleep=clock.sleep)  # 8 Mbit/s = 1 MB/s
        with tempfile.TemporaryDirectory() as directory:
            with (Path(directory) / "out").open("wb") as output:
                copy_limited(io.BytesIO(b"x" * 3_000_000), output, StorageBudget(Path(directory), None), throttle, chunk_size=500_000)
        self.assertAlmostEqual(clock.now, 3.0, places=5)
        self.assertEqual(Throttle(None).rate, None)


class AgentBudgetTests(unittest.TestCase):
    def test_tool_timeout_returns_failure(self):
        registry = ToolRegistry(); registry.register("slow", lambda: time.sleep(1) or "late")
        started = time.monotonic(); result = registry.execute(ToolCall("slow"), timeout_seconds=0.05)
        self.assertFalse(result.success); self.assertIn("timed out", result.output)
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertTrue(registry.execute(ToolCall("slow"), timeout_seconds=5).success)

    def test_agent_stops_on_time_and_token_budget(self):
        responses = ["plan", '{"tool": "calculator", "arguments": {"expression": "1 + 1"}}', '{"complete": false}'] * 3
        tools = ToolRegistry(); tools.register("calculator", calculator)
        clock = FakeClock()
        class SlowModel(ScriptedModelAdapter):
            def generate(self, messages): clock.now += 10; return super().generate(messages)
        result = AutonomousAgent(ModelRouter([SlowModel("slow", list(responses))]), tools, 3, AgentBudget(max_seconds=15), clock=clock).run("x")
        self.assertFalse(result.completed); self.assertIn("ngân sách", result.answer)
        self.assertTrue(any(item.startswith("budget:") for item in result.trace))
        wordy = ["một hai ba bốn năm sáu bảy tám chín mười"] + responses
        result = AutonomousAgent(ModelRouter([ScriptedModelAdapter("w", wordy)]), tools, 3, AgentBudget(max_generated_tokens=5)).run("x")
        self.assertIn("token limit", result.trace[-1])

    def test_agent_without_budget_still_completes(self):
        model = ScriptedModelAdapter("ok", ["plan", '{"tool": "calculator", "arguments": {"expression": "6 * 7"}}', '{"complete": true, "answer": "42"}'])
        tools = ToolRegistry(); tools.register("calculator", calculator)
        self.assertEqual(AutonomousAgent(ModelRouter([model]), tools, 2, AgentBudget(max_tool_seconds=5)).run("x").answer, "42")


if __name__ == "__main__": unittest.main()
