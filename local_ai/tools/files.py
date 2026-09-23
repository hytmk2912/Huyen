"""Công cụ đọc/ghi file cho agent, bị giới hạn trong một thư mục làm việc."""
from __future__ import annotations

from pathlib import Path

from local_ai.tools.core import ToolRegistry

MAX_FILE_BYTES = 200_000


class WorkspaceFiles:
    """Mọi đường dẫn được tính từ `root`; đường dẫn thoát ra ngoài (../, đường dẫn tuyệt đối khác) bị từ chối."""

    def __init__(self, root: str | Path, allow_write: bool = False, max_bytes: int = MAX_FILE_BYTES):
        self.root, self.allow_write, self.max_bytes = Path(root).resolve(), allow_write, max_bytes

    def _resolve(self, path: str) -> Path:
        target = (self.root / path).resolve()
        if target != self.root and self.root not in target.parents: raise ValueError(f"Path is outside the workspace: {path}")
        return target

    def read_file(self, path: str) -> str:
        target = self._resolve(path)
        if not target.is_file(): raise ValueError(f"File not found: {path}")
        if target.stat().st_size > self.max_bytes: raise ValueError(f"File is larger than {self.max_bytes} bytes: {path}")
        return target.read_text(encoding="utf-8", errors="replace")

    def list_files(self, path: str = ".") -> str:
        target = self._resolve(path)
        if not target.is_dir(): raise ValueError(f"Directory not found: {path}")
        return "\n".join(sorted(str(item.relative_to(self.root)) + ("/" if item.is_dir() else "") for item in target.iterdir()))

    def write_file(self, path: str, content: str) -> str:
        if not self.allow_write: raise ValueError("Writing files is disabled for this workspace")
        if len(content.encode("utf-8")) > self.max_bytes: raise ValueError(f"Content is larger than {self.max_bytes} bytes")
        target = self._resolve(path); target.parent.mkdir(parents=True, exist_ok=True); target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to {path}"


def register_file_tools(registry: ToolRegistry, root: str | Path, allow_write: bool = False) -> WorkspaceFiles:
    files = WorkspaceFiles(root, allow_write)
    registry.register("read_file", files.read_file); registry.register("list_files", files.list_files)
    if allow_write: registry.register("write_file", files.write_file)
    return files
