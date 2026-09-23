from __future__ import annotations

from typing import Protocol


class CodingTool(Protocol):
    def edit(self, path: str, content: str) -> str: ...


class ResearchTool(Protocol):
    def query(self, query: str) -> list[str]: ...


class TradingAnalysisTool(Protocol):
    """Analysis-only interface; execution/order placement is deliberately excluded."""
    def analyze(self, symbol: str, timeframe: str) -> str: ...


class FileTool(Protocol):
    def read(self, path: str) -> str: ...


class TerminalTool(Protocol):
    def run(self, command: str) -> str: ...
