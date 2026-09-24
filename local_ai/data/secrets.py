"""Quét repo tìm khóa/mật khẩu bị lộ trước khi commit (lệnh `python -m local_ai.data secret-scan`)."""
from __future__ import annotations

import re
from pathlib import Path

SECRET = re.compile(r"(?:hf_[A-Za-z0-9]{20,}|(?:api[_-]?key|password|secret)\s*[=:]\s*[^\s]{8,})", re.I)
SKIPPED_DIRS = {".git", ".venv", "venv", "__pycache__", ".runs", "storage"}
MAX_BYTES = 2_000_000


def scan_secrets(root: Path) -> list[str]:
    findings = []
    for path in root.rglob("*"):
        if not path.is_file() or path.name == "secrets.py" or SKIPPED_DIRS.intersection(path.parts):
            continue
        if path.stat().st_size >= MAX_BYTES:
            continue
        try:
            if SECRET.search(path.read_text(encoding="utf-8", errors="ignore")):
                findings.append(str(path))
        except OSError:
            pass
    return findings
