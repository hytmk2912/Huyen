"""Chạy các nhiệm vụ mẫu cho agent với công cụ calculator và TerminalTool, in trace và tỉ lệ thành công.

`python -m local_ai.agents.tasks --model ollama-colab` gọi model qua server kiểu OpenAI (adapter M3), ví dụ Ollama trên Colab.
`--scripted` dùng câu trả lời mẫu có sẵn trong file nhiệm vụ (không cần model) để thử cả chuỗi; `--dry-run` chỉ kiểm tra cấu hình.

Mỗi nhiệm vụ chạy trong một thư mục làm việc riêng, chép từ `data/eval/agent_workspace/`. TerminalTool chỉ được bật
trong thư mục đó, vẫn giữ mọi giới hạn của allowlist. Nhiệm vụ tính là đạt khi câu trả lời chứa đủ các chuỗi `expected`
và agent đã gọi thành công mọi công cụ trong `tools` (để chắc là model dùng công cụ chứ không đoán).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from local_ai.agents.loop import AutonomousAgent
from local_ai.config.settings import find_model_config
from local_ai.models.router import ModelRouter, ScriptedModelAdapter, create_adapter
from local_ai.runtime.terminal import TERMINAL_DESCRIPTION, TerminalConfig, TerminalTool
from local_ai.tools.core import CALCULATOR_DESCRIPTION, ToolRegistry, calculator

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASKS = ROOT / "data" / "eval" / "agent_tasks_v1.jsonl"
DEFAULT_WORKSPACE = ROOT / "data" / "eval" / "agent_workspace"
DEFAULT_MODELS = ROOT / "configs" / "models" / "platform.json"
TOOLS = ("calculator", "terminal")
THOUSANDS = re.compile(r"(?<=\d)[.,\s](?=\d{3}(?!\d))")


@dataclass(frozen=True)
class AgentTask:
    id: str
    prompt: str
    expected: tuple[str, ...]  # mọi chuỗi phải có trong câu trả lời (không phân biệt hoa thường, bỏ dấu phân cách hàng nghìn)
    tools: tuple[str, ...]  # công cụ agent phải gọi thành công ít nhất một lần
    reference: tuple[str, ...] = ()  # câu trả lời mẫu của model theo từng lượt hỏi, dùng cho --scripted


def load_tasks(path: str | Path = DEFAULT_TASKS) -> list[AgentTask]:
    tasks = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        value = json.loads(line)
        task = AgentTask(value["id"], value["prompt"], tuple(value["expected"]), tuple(value["tools"]), tuple(value.get("reference", ())))
        unknown = set(task.tools) - set(TOOLS)
        if unknown or not task.expected: raise ValueError(f"Nhiệm vụ '{task.id}': công cụ không có ({', '.join(sorted(unknown))}) hoặc thiếu expected")
        tasks.append(task)
    if len({task.id for task in tasks}) != len(tasks): raise ValueError(f"File nhiệm vụ {path} có id bị lặp")
    return tasks


def normalize_answer(text: str) -> str:
    return THOUSANDS.sub("", " ".join(str(text).casefold().split()))


def answer_matches(answer: str, expected: tuple[str, ...]) -> bool:
    return all(normalize_answer(item) in normalize_answer(answer) for item in expected)


def build_tools(workspace: Path, log_path: Path, used: list[str]) -> ToolRegistry:
    """calculator và TerminalTool (bật, chỉ trong `workspace`); ghi tên công cụ mỗi lần gọi thành công vào `used`."""
    def tracked(name: str, tool: Callable[..., str]) -> Callable[..., str]:
        def call(**arguments: Any) -> str:
            output = tool(**arguments); used.append(name); return output
        return call
    terminal = TerminalTool(replace(TerminalConfig.from_file(), workspace=workspace, log_path=log_path), enabled=True)
    registry = ToolRegistry()
    registry.register("calculator", tracked("calculator", calculator), CALCULATOR_DESCRIPTION)
    registry.register("terminal", tracked("terminal", terminal), TERMINAL_DESCRIPTION)
    return registry


def prepare_workspace(root: Path, task: AgentTask, source: Path = DEFAULT_WORKSPACE) -> tuple[Path, Path]:
    """Thư mục làm việc mới cho một nhiệm vụ (chép từ data/eval/agent_workspace/) và file log của TerminalTool (nằm ngoài thư mục đó)."""
    base = root / task.id; workspace = base / "workspace"
    shutil.rmtree(base, ignore_errors=True); shutil.copytree(source, workspace)
    return workspace, base / "terminal.jsonl"


def run_task(task: AgentTask, model: Any, root: Path, max_iterations: int) -> dict[str, Any]:
    workspace, log_path = prepare_workspace(root, task)
    used: list[str] = []
    result = AutonomousAgent(ModelRouter([model], default=model.name), build_tools(workspace, log_path, used), max_iterations).run(task.prompt)
    missing = [tool for tool in task.tools if tool not in used]
    if not result.completed: reason = "agent không hoàn thành"
    elif not answer_matches(result.answer, task.expected): reason = "câu trả lời thiếu " + ", ".join(task.expected)
    elif missing: reason = "không gọi công cụ bắt buộc: " + ", ".join(missing)
    else: reason = None
    return {"id": task.id, "prompt": task.prompt, "success": reason is None, "reason": reason, "answer": result.answer, "tools_used": used, "trace": list(result.trace)}


def run_tasks(tasks: list[AgentTask], model_factory: Callable[[AgentTask], Any], root: str | Path, max_iterations: int = 4) -> dict[str, Any]:
    """Chạy lần lượt từng nhiệm vụ; `model_factory(task)` trả adapter cho nhiệm vụ đó (model thật dùng chung một adapter)."""
    start, results = time.monotonic(), []
    for task in tasks:
        began = time.monotonic()
        results.append({**run_task(task, model_factory(task), Path(root), max_iterations), "duration_s": round(time.monotonic() - began, 2)})
    passed = sum(item["success"] for item in results)
    return {"status": "completed", "tasks": results, "passed": passed, "total": len(results), "success_rate": round(passed / len(results), 4) if results else 0.0,
            "duration_s": round(time.monotonic() - start, 2)}


def format_report(report: dict[str, Any], model: str, width: int = 400) -> str:
    lines = [f"Agent với model {model}: {report['total']} nhiệm vụ", ""]
    for item in report["tasks"]:
        lines.append(f"=== {item['id']}: {'ĐẠT' if item['success'] else 'KHÔNG ĐẠT'} ===")
        lines.append(f"Yêu cầu: {item['prompt']}")
        for step in item["trace"]:
            step = step if len(step) <= width else step[:width] + "…"
            lines.append("  " + step.rstrip("\n").replace("\n", "\n    "))  # output nhiều dòng (ls, cat) thụt vào cho dễ đọc
        lines.append(f"Trả lời: {item['answer']}")
        lines.append(f"Công cụ đã gọi: {', '.join(item['tools_used']) or 'không có'}" + (f" | Lý do: {item['reason']}" if item["reason"] else ""))
        lines.append("")
    lines.append(f"Tỉ lệ thành công: {report['passed']}/{report['total']} ({report['success_rate']:.0%}); thời gian: {report['duration_s'] / 60:.1f} phút")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m local_ai.agents.tasks", description="Chạy các nhiệm vụ mẫu cho agent (calculator và TerminalTool), in trace và tỉ lệ thành công.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--model", help="Tên model trong danh sách model, ví dụ ollama-colab (server kiểu OpenAI)")
    target.add_argument("--scripted", action="store_true", help="Dùng câu trả lời mẫu trong file nhiệm vụ thay cho model (không cần server)")
    parser.add_argument("--models", default=str(DEFAULT_MODELS), help="File danh sách model")
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS), help="File nhiệm vụ (JSONL)")
    parser.add_argument("--workspace", default=".runs/agent_tasks", help="Thư mục chứa thư mục làm việc của từng nhiệm vụ")
    parser.add_argument("--max-iterations", type=int, default=4, help="Số lần agent được gọi công cụ cho mỗi nhiệm vụ")
    parser.add_argument("--output", help="Ghi kết quả chi tiết (JSON) vào file này")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ kiểm tra file nhiệm vụ và cấu hình model, không gọi model")
    args = parser.parse_args(argv)
    tasks = load_tasks(args.tasks)
    config = None if args.scripted else find_model_config(args.models, args.model)
    if config is not None and not {"reasoning"} & {capability.value for capability in config.capabilities}:
        print(f"Model '{config.name}' cần khả năng reasoning để làm agent", file=sys.stderr); return 2
    if args.dry_run:
        model = {"name": "scripted"} if config is None else {"name": config.name, "backend": config.backend, "base_url": config.base_url, "source": config.source, "timeout_s": config.timeout_s}
        print(json.dumps({"status": "dry-run", "model": model, "tasks": [{"id": task.id, "tools": list(task.tools)} for task in tasks], "workspace": args.workspace, "max_iterations": args.max_iterations}, ensure_ascii=False, indent=2))
        return 0
    if config is None:
        factory = lambda task: ScriptedModelAdapter("scripted", list(task.reference))
        name = "scripted (câu trả lời mẫu)"
    else:
        shared = create_adapter(config); factory = lambda task: shared; name = f"{config.name} ({config.source})"
    report = run_tasks(tasks, factory, args.workspace, args.max_iterations)
    print(format_report(report, name))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps({**report, "model": name}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
