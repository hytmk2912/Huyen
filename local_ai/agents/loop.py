from __future__ import annotations

import json
import re
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


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCED_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json_object(text: str) -> dict[str, Any]:
    """Tách object JSON đầu tiên trong câu trả lời của model.

    Chịu được: chữ thừa trước/sau, khối ```json ... ``` hoặc ``` ... ```, khối <think>...</think>
    (JSON nằm trong <think> bị bỏ qua) và dấu { } nằm trong chuỗi. Không tìm thấy thì trả về {}.
    """
    text = _THINK_BLOCK.sub("", text or "")
    if "<think>" in text.lower():  # <think> chưa đóng: phần sau nó chưa phải câu trả lời
        text = text[: text.lower().index("<think>")]
    decoder = json.JSONDecoder()
    candidates = [block.strip() for block in _FENCED_BLOCK.findall(text)] + [text.strip()]
    for candidate in candidates:
        for start in (index for index, char in enumerate(candidate) if char == "{"):
            try:
                value, _ = decoder.raw_decode(candidate, start)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    return {}


# Giữ tên cũ để code khác không bị ảnh hưởng.
_parse_json_object = extract_json_object


class _ModelFailure(Exception):
    """Model (ví dụ server local) lỗi khi sinh câu trả lời; agent dừng êm thay vì sập."""


class AutonomousAgent:
    """Vòng lặp có giới hạn: lập kế hoạch → hành động → quan sát → đánh giá → sửa, do các model trong cấu hình điều khiển."""

    def __init__(self, router: ModelRouter, tools: ToolRegistry, max_iterations: int = 3):
        self.router, self.tools, self.max_iterations = router, tools, max_iterations

    def run(self, request: str) -> AgentResult:
        trace = [f"request: {request}"]
        try:
            return self._run(request, trace)
        except _ModelFailure as error:
            trace.append(f"model error: {error}")
            return AgentResult(f"Model lỗi, agent dừng lại: {error}", tuple(trace), False)

    @staticmethod
    def _ask(model, content: str) -> str:
        try:
            return model.generate([Message("user", content)])
        except Exception as error:  # server chưa chạy, hết thời gian chờ, lỗi HTTP... không làm sập agent
            raise _ModelFailure(f"{type(error).__name__}: {error}") from error

    def _run(self, request: str, trace: list[str]) -> AgentResult:
        planner = self.router.select("reasoning")
        plan = self._ask(planner, f"Plan this task: {request}")
        trace.append(f"plan: {plan}")
        observation = ""
        for attempt in range(self.max_iterations):
            decision = _parse_json_object(self._ask(planner, json.dumps({
                "request": request, "plan": plan, "observation": observation, "attempt": attempt,
                "instruction": "Return JSON with tool, arguments, and expected fields.",
            })))
            if not isinstance(decision.get("tool"), str):
                observation = "Invalid decision: the model must return JSON with a tool name."
                trace.append(f"error: {observation}")
                continue
            result = self.tools.execute(ToolCall(decision["tool"], decision.get("arguments", {})))
            observation = result.output
            trace.append(f"tool[{result.name}]: {observation}")
            verdict = self._ask(planner, json.dumps({
                "request": request, "observation": observation,
                "instruction": "Return JSON with complete boolean and answer or correction.",
            }))
            evaluation = _parse_json_object(verdict)
            trace.append(f"evaluation: {verdict}")
            if evaluation.get("complete") and result.success and "answer" in evaluation:
                return AgentResult(str(evaluation["answer"]), tuple(trace), True)
            plan = evaluation.get("correction", plan)
        return AgentResult("Không hoàn thành được nhiệm vụ trong số lần thử lại cho phép.", tuple(trace), False)
