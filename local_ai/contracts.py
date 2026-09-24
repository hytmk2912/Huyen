from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Message:
    role: str
    content: str
    images: tuple[str, ...] = ()  # đường dẫn file ảnh cục bộ, chỉ model multimodal nhận được


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    name: str
    output: str
    success: bool


class ModelAdapter(Protocol):
    name: str
    capabilities: set[str]

    def generate(self, messages: list[Message]) -> str: ...


class Retriever(Protocol):
    def search(self, query: str, limit: int = 5) -> list[str]: ...
