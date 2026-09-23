from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable

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


@dataclass(frozen=True)
class AgentBudget:
    """Ngân sách cho một nhiệm vụ; None là không giới hạn. Token được đếm bằng token_counter (mặc định ước lượng theo số từ)."""
    max_seconds: float | None = None
    max_tool_seconds: float | None = None
    max_generated_tokens: int | None = None


class _BudgetTracker:
    def __init__(self, budget: AgentBudget, token_counter: Callable[[str], int], clock: Callable[[], float]):
        self.budget, self.count, self.clock, self.start, self.tokens = budget, token_counter, clock, clock(), 0

    def record(self, text: str) -> None: self.tokens += self.count(text)

    def exceeded(self) -> str | None:
        if self.budget.max_seconds is not None and self.clock() - self.start > self.budget.max_seconds: return f"time limit of {self.budget.max_seconds} seconds"
        if self.budget.max_generated_tokens is not None and self.tokens > self.budget.max_generated_tokens: return f"token limit of {self.budget.max_generated_tokens}"
        return None


def _safe_generate(model, messages: list[Message], trace: list[str]) -> str:
    """Phục hồi khi model lỗi (mất kết nối, hết bộ nhớ...): ghi lỗi vào trace và trả về chuỗi rỗng."""
    try:
        return model.generate(messages)
    except Exception as error:
        trace.append(f"error: model failure: {type(error).__name__}: {error}")
        return ""


class AutonomousAgent:
    """Vòng lặp có giới hạn: lập kế hoạch → hành động → quan sát → đánh giá → sửa, do các model trong cấu hình điều khiển."""

    def __init__(self, router: ModelRouter, tools: ToolRegistry, max_iterations: int = 3, budget: AgentBudget | None = None, token_counter: Callable[[str], int] | None = None, clock: Callable[[], float] = time.monotonic):
        self.router, self.tools, self.max_iterations = router, tools, max_iterations
        self.budget, self.token_counter, self.clock = budget or AgentBudget(), token_counter or (lambda text: len(text.split())), clock

    def _stopped(self, tracker: _BudgetTracker, trace: list[str]) -> AgentResult | None:
        reason = tracker.exceeded()
        if reason is None: return None
        trace.append(f"budget: stopped after exceeding the {reason}")
        return AgentResult(f"Đã dừng vì vượt ngân sách ({reason}).", tuple(trace), False)

    def run(self, request: str) -> AgentResult:
        planner = self.router.select("reasoning")
        trace = [f"request: {request}"]
        tracker = _BudgetTracker(self.budget, self.token_counter, self.clock)

        def generate(content: str) -> str:
            text = _safe_generate(planner, [Message("user", content)], trace); tracker.record(text); return text

        plan = generate(f"Plan this task: {request}")
        trace.append(f"plan: {plan}")
        observation = ""
        for attempt in range(self.max_iterations):
            if stopped := self._stopped(tracker, trace): return stopped
            decision = _parse_json_object(generate(json.dumps({
                "request": request, "plan": plan, "observation": observation, "attempt": attempt,
                "instruction": "Return JSON with tool, arguments, and expected fields.",
            })))
            if not isinstance(decision.get("tool"), str):
                observation = "Invalid decision: the model must return JSON with a tool name."
                trace.append(f"error: {observation}")
                continue
            if stopped := self._stopped(tracker, trace): return stopped
            result = self.tools.execute(ToolCall(decision["tool"], decision.get("arguments", {})), self.budget.max_tool_seconds)
            observation = result.output
            trace.append(f"tool[{result.name}]: {observation}")
            verdict = generate(json.dumps({
                "request": request, "observation": observation,
                "instruction": "Return JSON with complete boolean and answer or correction.",
            }))
            evaluation = _parse_json_object(verdict)
            trace.append(f"evaluation: {verdict}")
            if evaluation.get("complete") and result.success and "answer" in evaluation:
                return AgentResult(str(evaluation["answer"]), tuple(trace), True)
            plan = evaluation.get("correction", plan)
        return AgentResult("Không hoàn thành được nhiệm vụ trong số lần thử lại cho phép.", tuple(trace), False)
