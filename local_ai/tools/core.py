from __future__ import annotations

import ast
import operator
from typing import Any, Callable

from local_ai.contracts import ToolCall, ToolResult


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., str]] = {}
        self._descriptions: dict[str, str] = {}

    def register(self, name: str, tool: Callable[..., str], description: str = "") -> None:
        """`description` (tiếng Anh, vì gửi cho model) nói công cụ làm gì và nhận tham số nào; agent đưa nó vào prompt."""
        if name in self._tools:
            raise ValueError(f"Công cụ đã được đăng ký: {name}")
        self._tools[name] = tool
        self._descriptions[name] = description

    def describe(self) -> dict[str, str]:
        """Tên công cụ -> mô tả, để model biết có những công cụ nào và gọi thế nào."""
        return dict(self._descriptions)

    def execute(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(call.name, "Unknown tool", False)
        try:
            return ToolResult(call.name, str(tool(**call.arguments)), True)
        except Exception as error:  # mọi lỗi của công cụ (chia 0, sai cú pháp...) thành kết quả thất bại, agent không sập
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


CALCULATOR_DESCRIPTION = 'Evaluate an arithmetic expression with numbers, + - * / and parentheses. Arguments: {"expression": "(17 * 23) + 158"}'
