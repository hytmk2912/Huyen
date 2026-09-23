import json
import tempfile
import unittest
from pathlib import Path

from local_ai.agents.loop import AutonomousAgent
from local_ai.config.settings import load_settings
from local_ai.contracts import ToolCall
from local_ai.data.manifest import DatasetManifest
from local_ai.demo import run_demo
from local_ai.evaluation.benchmarks import BenchmarkCase, evaluate
from local_ai.experiments.tracking import RunTracker, seed_everything
from local_ai.memory.context import MemoryStore
from local_ai.models.router import ModelRouter, ScriptedModelAdapter
from local_ai.tools.core import ToolRegistry, calculator
from local_ai.tools.sandbox import PythonSandbox
from local_ai.training.plans import TrainingPlan


class FoundationTests(unittest.TestCase):
    def test_config_and_router(self):
        settings = load_settings("configs/demo.json")
        model = ScriptedModelAdapter("a", ["ok"], {"reasoning"})
        self.assertEqual(settings.seed, 7)
        self.assertEqual(ModelRouter([model]).select("reasoning").name, "a")
        with self.assertRaises(LookupError):
            ModelRouter([model]).select("planning")

    def test_tools_and_sandbox(self):
        registry = ToolRegistry()
        registry.register("calculator", calculator)
        self.assertEqual(registry.execute(ToolCall("calculator", {"expression": "2 + 3 * 4"})).output, "14")
        self.assertFalse(registry.execute(ToolCall("missing")).success)
        result = PythonSandbox().run("print('isolated')")
        self.assertEqual(result.stdout.strip(), "isolated")
        self.assertEqual(result.returncode, 0)

    def test_memory_dataset_training_tracking_and_evaluation(self):
        memory = MemoryStore(1)
        from local_ai.contracts import Message
        memory.add(Message("user", "first")); memory.add(Message("assistant", "second"))
        self.assertEqual([message.content for message in memory.context()], ["second"])
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "dataset.json"
            DatasetManifest("demo", "1", "local", "train").save(manifest)
            self.assertEqual(json.loads(manifest.read_text())["name"], "demo")
            tracker = RunTracker(Path(directory) / "run", {"seed": 1})
            tracker.record_metrics({"accuracy": 1.0}); tracker.record_checkpoint("checkpoint-1")
            self.assertTrue((Path(directory) / "run" / "metrics.json").exists())
        seed_everything(12)
        plan = TrainingPlan("sft", "manifest.json", "base", 12, "out")
        plan.validate()
        self.assertEqual(evaluate([BenchmarkCase("one", "x", "y")], lambda _: "y")["accuracy"], 1.0)

    def test_agent_end_to_end_demo(self):
        self.assertEqual(run_demo(), "The result is 42.")

    def test_agent_corrects_and_retries(self):
        model = ScriptedModelAdapter("retry", [
            "Use the calculator.",
            '{"tool": "missing", "arguments": {}}',
            '{"complete": false, "correction": "Use calculator instead."}',
            '{"tool": "calculator", "arguments": {"expression": "6 * 7"}}',
            '{"complete": true, "answer": "42"}',
        ])
        tools = ToolRegistry(); tools.register("calculator", calculator)
        result = AutonomousAgent(ModelRouter([model]), tools, 2).run("Calculate 6 * 7")
        self.assertTrue(result.completed)
        self.assertEqual(result.answer, "42")
        self.assertTrue(any("Unknown tool" in item for item in result.trace))


if __name__ == "__main__":
    unittest.main()

class ModelConfigurationTests(unittest.TestCase):
    def test_primary_model_configuration_and_capability_routing(self):
        from local_ai.config.settings import load_model_configs
        from local_ai.models.adapters import ModelCapability
        configs = load_model_configs("configs/models/platform.json")
        self.assertEqual(configs[0].source, "huihui-ai/Huihui-Qwen3.8-27B-abliterated")
        self.assertIn(ModelCapability.CODING, configs[0].capabilities)
        self.assertTrue(configs[0].configuration_hash)

    def test_agent_runs_on_models_built_from_platform_config(self):
        from local_ai.config.settings import load_model_configs
        responses = [
            "Use the calculator.",
            '{"tool": "calculator", "arguments": {"expression": "6 * 7"}}',
            '{"complete": true, "answer": "42"}',
        ]
        router = ModelRouter.from_configs(
            load_model_configs("configs/models/platform.json"),
            lambda config: ScriptedModelAdapter(config.name, responses, {item.value for item in config.capabilities}),
        )
        tools = ToolRegistry(); tools.register("calculator", calculator)
        result = AutonomousAgent(router, tools, 2).run("Calculate 6 * 7")
        self.assertTrue(result.completed)
        self.assertEqual(result.answer, "42")

    def test_agent_survives_malformed_model_json(self):
        model = ScriptedModelAdapter("bad-json", ["plan", "not json", "still not json"])
        result = AutonomousAgent(ModelRouter([model]), ToolRegistry(), 2).run("anything")
        self.assertFalse(result.completed)
        self.assertTrue(any(item.startswith("error:") for item in result.trace))
