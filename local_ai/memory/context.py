from __future__ import annotations

from collections import deque
from local_ai.contracts import Message


class MemoryStore:
    def __init__(self, capacity: int = 50):
        self._messages: deque[Message] = deque(maxlen=capacity)

    def add(self, message: Message) -> None:
        self._messages.append(message)

    def context(self) -> list[Message]:
        return list(self._messages)
