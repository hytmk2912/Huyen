from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

VALID_DOMAINS = {
    "coding",
    "software_engineering",
    "math_logic",
    "reasoning",
    "planning",
    "tool_use",
    "agent",
    "research",
    "trading",
    "debugging",
}
VALID_STATUSES = {"pending", "valid", "rejected"}


@dataclass(frozen=True)
class DatasetExample:
    id: str
    domain: str
    task: str
    input: str
    expected_output: str
    source: dict[str, str]
    license: dict[str, str]
    dataset_version: str
    context: str | None = None
    reasoning: str | None = None
    # Hội thoại đầy đủ (kể cả system và nhiều lượt); khi có thì sft.jsonl dùng nguyên mảng này.
    messages: list[dict[str, str]] | None = None
    trajectory: list[dict[str, Any]] | None = None
    tools_used: list[str] = field(default_factory=list)
    difficulty: int = 1
    quality_score: float | None = None
    validation_status: Literal["pending", "valid", "rejected"] = "pending"
    verification: dict[str, Any] = field(default_factory=dict)
    split: Literal["train", "eval"] = "train"
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
