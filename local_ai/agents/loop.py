from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from local_ai.contracts import Message, ToolCall
from local_ai.models.router import ModelRouter
from local_ai.tools.core import ToolRegistry


@dataclass(frozen=True)
class AgentResult:
    answer: str
    trace: tuple[str, ...]
    completed: bool


def _parse_json_object(text: str) -> dict[str, Any]:
    """Model thật hay trả về JSON lỗi; coi đó là câu trả lời rỗng thay vì làm chương trình dừng."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _safe_generate(model, messages: list[Message], trace: list[str]) -> str:
    """Phục hồi khi model lỗi (mất kết nối, hết bộ nhớ...): ghi lỗi vào trace và trả về chuỗi rỗng."""
    try:
        return model.generate(messages)
    except Exception as error:
        trace.append(f"error: model failure: {type(error).__name__}: {error}")
        return ""


class AutonomousAgent:
    """Vòng lặp có giới hạn: lập kế hoạch → hành động → quan sát → đánh giá → sửa, do các model trong cấu hình điều khiển."""

    def __init__(self, router: ModelRouter, tools: ToolRegistry, max_iterations: int = 3):
        self.router, self.tools, self.max_iterations = router, tools, max_iterations

    def run(self, request: str) -> AgentResult:
        planner = self.router.select("reasoning")
        trace = [f"request: {request}"]
        plan = _safe_generate(planner, [Message("user", f"Plan this task: {request}")], trace)
        trace.append(f"plan: {plan}")
        observation = ""
        for attempt in range(self.max_iterations):
            decision = _parse_json_object(_safe_generate(planner, [Message("user", json.dumps({
                "request": request, "plan": plan, "observation": observation, "attempt": attempt,
                "instruction": "Return JSON with tool, arguments, and expected fields.",
            }))], trace))
            if not isinstance(decision.get("tool"), str):
                observation = "Invalid decision: the model must return JSON with a tool name."
                trace.append(f"error: {observation}")
                continue
            result = self.tools.execute(ToolCall(decision["tool"], decision.get("arguments", {})))
            observation = result.output
            trace.append(f"tool[{result.name}]: {observation}")
            verdict = _safe_generate(planner, [Message("user", json.dumps({
                "request": request, "observation": observation,
                "instruction": "Return JSON with complete boolean and answer or correction.",
            }))], trace)
            evaluation = _parse_json_object(verdict)
            trace.append(f"evaluation: {verdict}")
            if evaluation.get("complete") and result.success and "answer" in evaluation:
                return AgentResult(str(evaluation["answer"]), tuple(trace), True)
            plan = evaluation.get("correction", plan)
        return AgentResult("Không hoàn thành được nhiệm vụ trong số lần thử lại cho phép.", tuple(trace), False)
