from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


BENCHMARK_SUITES = ("coding", "reasoning", "tool_use", "planning", "trading", "autonomous_task_completion")


@dataclass(frozen=True)
class BenchmarkCase:
    identifier: str
    prompt: str
    expected: str


def evaluate(cases: list[BenchmarkCase], runner: Callable[[str], str]) -> dict[str, float]:
    correct = sum(runner(case.prompt).strip() == case.expected.strip() for case in cases)
    return {"cases": float(len(cases)), "accuracy": correct / len(cases) if cases else 0.0}
