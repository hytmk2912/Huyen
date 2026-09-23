from __future__ import annotations

import ast
import concurrent.futures
import operator
from typing import Any, Callable

from local_ai.contracts import ToolCall, ToolResult


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., str]] = {}

    def register(self, name: str, tool: Callable[..., str]) -> None:
        if name in self._tools:
            raise ValueError(f"Công cụ đã được đăng ký: {name}")
        self._tools[name] = tool

    def execute(self, call: ToolCall, timeout_seconds: float | None = None) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(call.name, "Unknown tool", False)
        try:
            if timeout_seconds is None:
                return ToolResult(call.name, str(tool(**call.arguments)), True)
            # Chạy trong luồng riêng để trả về đúng hạn; công cụ chạy quá hạn bị bỏ qua kết quả.
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                return ToolResult(call.name, str(executor.submit(tool, **call.arguments).result(timeout=timeout_seconds)), True)
            except concurrent.futures.TimeoutError:
                return ToolResult(call.name, f"Tool timed out after {timeout_seconds} seconds", False)
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
        except (TypeError, ValueError) as error:
            return ToolResult(call.name, str(error), False)
        except Exception as error:  # phục hồi: lỗi bất ngờ của công cụ không được làm dừng agent
            return ToolResult(call.name, f"Tool error: {type(error).__name__}: {error}", False)


_ALLOWED_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def calculator(expression: str) -> str:
    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
            return _ALLOWED_OPERATORS[type(node.op)](evaluate(node.left), evaluate(node.right))
        raise ValueError("Only numeric arithmetic expressions are allowed")

    return str(evaluate(ast.parse(expression, mode="eval").body))
