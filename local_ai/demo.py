"""Demo agent + công cụ máy tính.

Mặc định dùng model giả (tất định, không cần mạng). Thêm `--model <tên>` để chạy qua một model trong
danh sách model, ví dụ server local kiểu OpenAI: `python -m local_ai.demo --model ollama`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from local_ai.agents.loop import AutonomousAgent
from local_ai.config.settings import find_model_config, load_settings
from local_ai.models.router import ModelRouter, ScriptedModelAdapter, create_adapter
from local_ai.tools.core import CALCULATOR_DESCRIPTION, ToolRegistry, calculator

ROOT = Path(__file__).parent.parent


def run_demo(model_name: str | None = None, models_path: str | Path = ROOT / "configs" / "models" / "platform.json", request: str = "What is 21 * 2?") -> str:
    settings = load_settings(ROOT / "configs" / "demo.json")
    if model_name is None:
        model = ScriptedModelAdapter("demo-scripted", [
            "Calculate the requested arithmetic expression.",
            '{"tool": "calculator", "arguments": {"expression": "21 * 2"}, "expected": "42"}',
            '{"complete": true, "answer": "The result is 42."}',
        ])
    else:
        model = create_adapter(find_model_config(models_path, model_name))
    tools = ToolRegistry()
    tools.register("calculator", calculator, CALCULATOR_DESCRIPTION)
    result = AutonomousAgent(ModelRouter([model], default=model.name), tools, settings.max_iterations).run(request)
    if not result.completed:
        raise RuntimeError(result.answer)
    return result.answer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m local_ai.demo", description="Chạy agent demo với công cụ máy tính.")
    parser.add_argument("--model", help="Tên model trong danh sách model (ví dụ ollama, llamacpp); bỏ trống thì dùng model giả")
    parser.add_argument("--models", default=str(ROOT / "configs" / "models" / "platform.json"), help="File danh sách model")
    parser.add_argument("--request", default="What is 21 * 2?", help="Yêu cầu gửi cho agent")
    args = parser.parse_args(argv)
    try:
        print(run_demo(args.model, args.models, args.request))
    except RuntimeError as error:
        print(f"Demo không hoàn thành: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
