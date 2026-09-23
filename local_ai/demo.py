from __future__ import annotations

from pathlib import Path

from local_ai.agents.loop import AutonomousAgent
from local_ai.config.settings import load_settings
from local_ai.models.router import ModelRouter, ScriptedModelAdapter
from local_ai.tools.core import ToolRegistry, calculator


def run_demo() -> str:
    settings = load_settings(Path(__file__).parent.parent / "configs" / "demo.json")
    model = ScriptedModelAdapter("demo-scripted", [
        "Calculate the requested arithmetic expression.",
        '{"tool": "calculator", "arguments": {"expression": "21 * 2"}, "expected": "42"}',
        '{"complete": true, "answer": "The result is 42."}',
    ])
    tools = ToolRegistry()
    tools.register("calculator", calculator)
    result = AutonomousAgent(ModelRouter([model]), tools, settings.max_iterations).run("What is 21 * 2?")
    if not result.completed:
        raise RuntimeError(result.answer)
    return result.answer


if __name__ == "__main__":
    print(run_demo())
