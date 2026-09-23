"""Giới hạn tài nguyên khi thu thập corpus: dung lượng ổ đĩa, băng thông, và tải tiếp sau khi dừng."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, BinaryIO, Callable


class StorageLimitReached(RuntimeError):
    """Dừng an toàn: phần đã tải được giữ lại trong file .part để lần sau tải tiếp."""


def storage_used_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


class StorageBudget:
    def __init__(self, root: Path, max_gb: float | None):
        self.root, self.limit = root, int(max_gb * 1024 ** 3) if max_gb else None
        self.used = storage_used_bytes(root) if self.limit else 0

    def reserve(self, nbytes: int) -> None:
        if self.limit is None: return
        if self.used + nbytes > self.limit:
            raise StorageLimitReached(f"Vượt giới hạn ổ đĩa max_local_storage_gb ({self.limit / 1024 ** 3:.2f} GB); đã dùng {self.used / 1024 ** 3:.2f} GB")
        self.used += nbytes


class Throttle:
    """Giữ tốc độ trung bình không vượt max_mbps (megabit/giây) bằng cách ngủ khi tải quá nhanh."""
    def __init__(self, max_mbps: float | None, clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        self.rate = max_mbps * 1_000_000 / 8 if max_mbps else None
        self.clock, self.sleep, self.start, self.sent = clock, sleep, clock(), 0

    def consume(self, nbytes: int) -> None:
        if not self.rate: return
        self.sent += nbytes
        wait = self.sent / self.rate - (self.clock() - self.start)
        if wait > 0: self.sleep(wait)


def limits_from_config(config: dict[str, Any]) -> tuple[StorageBudget, Throttle]:
    return StorageBudget(Path(config["storage_root"]), config.get("max_local_storage_gb")), Throttle(config.get("max_bandwidth_mbps"))


def copy_limited(source: BinaryIO, output: BinaryIO, budget: StorageBudget, throttle: Throttle, chunk_size: int = 1024 * 1024) -> int:
    written = 0
    while chunk := source.read(chunk_size):
        budget.reserve(len(chunk)); output.write(chunk); output.flush(); throttle.consume(len(chunk)); written += len(chunk)
    return written
