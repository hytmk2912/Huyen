from __future__ import annotations

import json
from dataclasses import dataclass

from local_ai.contracts import Message, ToolCall
from local_ai.models.router import ModelRouter
from local_ai.tools.core import ToolRegistry


@dataclass(frozen=True)
class AgentResult:
    answer: str
    trace: tuple[str, ...]
    completed: bool


class AutonomousAgent:
    """Bounded plan-act-observe-evaluate-correct loop driven by configured models."""

    def __init__(self, router: ModelRouter, tools: ToolRegistry, max_iterations: int = 3):
        self.router, self.tools, self.max_iterations = router, tools, max_iterations

    def run(self, request: str) -> AgentResult:
        planner = self.router.select("planning")
        trace = [f"request: {request}"]
        plan = planner.generate([Message("user", f"Plan this task: {request}")])
        trace.append(f"plan: {plan}")
        observation = ""
        for attempt in range(self.max_iterations):
            decision = json.loads(planner.generate([Message("user", json.dumps({
                "request": request, "plan": plan, "observation": observation, "attempt": attempt,
                "instruction": "Return JSON with tool, arguments, and expected fields.",
            }))]))
            result = self.tools.execute(ToolCall(decision["tool"], decision.get("arguments", {})))
            observation = result.output
            trace.append(f"tool[{result.name}]: {observation}")
            verdict = planner.generate([Message("user", json.dumps({
                "request": request, "observation": observation,
                "instruction": "Return JSON with complete boolean and answer or correction.",
            }))])
            evaluation = json.loads(verdict)
            trace.append(f"evaluation: {verdict}")
            if evaluation.get("complete") and result.success:
                return AgentResult(evaluation["answer"], tuple(trace), True)
            plan = evaluation.get("correction", plan)
        return AgentResult("Unable to complete task within the configured retry limit.", tuple(trace), False)
