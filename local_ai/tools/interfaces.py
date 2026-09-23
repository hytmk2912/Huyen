from __future__ import annotations

from typing import Protocol


class CodingTool(Protocol):
    def edit(self, path: str, content: str) -> str: ...


class ResearchTool(Protocol):
    def query(self, query: str) -> list[str]: ...


class TradingAnalysisTool(Protocol):
    """Giao diện chỉ để phân tích; cố ý không có chức năng thực hiện/đặt lệnh."""
    def analyze(self, symbol: str, timeframe: str) -> str: ...


class FileTool(Protocol):
    def read(self, path: str) -> str: ...


class TerminalTool(Protocol):
    def run(self, command: str) -> str: ...
