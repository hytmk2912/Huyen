"""Quét repo tìm khóa/mật khẩu bị lộ (lệnh `python -m local_ai.data secret-scan`)."""
from __future__ import annotations

import re
from pathlib import Path

SECRET = re.compile(r"(?:hf_[A-Za-z0-9]{20,}|(?:api[_-]?key|password|secret)\s*[=:]\s*[^\s]{8,})", re.I)
SKIPPED_DIRS = {".git", ".venv", "venv", "__pycache__"}
MAX_BYTES = 2_000_000


def scan_secrets(root: Path) -> list[str]:
    """Trả về danh sách file có chuỗi giống token Hugging Face hoặc khóa/mật khẩu viết thẳng trong code."""
    findings = []
    for path in root.rglob("*"):
        if not path.is_file() or SKIPPED_DIRS.intersection(path.parts):
            continue
        if path.stat().st_size >= MAX_BYTES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        # Bỏ qua chính dòng định nghĩa mẫu tìm kiếm (SECRET = re.compile(...)), không phải khóa thật.
        if any("re.compile(" not in match.group(0) for match in SECRET.finditer(text)):
            findings.append(str(path))
    return findings
