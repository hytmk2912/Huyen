"""Quét repo tìm khóa/mật khẩu bị lộ (lệnh `python -m local_ai.data secret-scan`)."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

# Biến shell viết HOA có chữ PASS gán giá trị viết thẳng, kể cả giá trị mặc định kiểu SYNC_PASS="${SYNC_PASS:-...}"
# (mật khẩu rsync từng lọt vào lịch sử commit theo đúng kiểu này). Giá trị bắt đầu bằng $ là lấy từ biến khác nên bỏ qua.
SECRET = re.compile(r"(?:hf_[A-Za-z0-9]{20,}|(?:api[_-]?key|password|secret)\s*[=:]\s*[^\s]{8,}|(?-i:\b[A-Z_]*PASS(?:WORD|WD)?[A-Z_]*=)[\"']?(?:\$\{[A-Za-z_]+:-)?[^\s\"'$}]{8,})", re.I)
SKIPPED_DIRS = {".git", ".venv", "venv", "__pycache__"}
MAX_BYTES = 2_000_000


def candidate_files(root: Path) -> list[Path]:
    """File có thể bị commit: file git đang theo dõi và file mới chưa bị .gitignore loại (dữ liệu tải về, .runs/, .env không bị quét).
    Không phải repo git thì quét mọi file, trừ các thư mục trong SKIPPED_DIRS."""
    try:
        listed = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return [path for path in root.rglob("*") if not SKIPPED_DIRS.intersection(path.parts)]
    return [root / name for name in listed.decode("utf-8", errors="surrogateescape").split("\0") if name]


def scan_secrets(root: Path) -> list[str]:
    """Trả về danh sách file có chuỗi giống token Hugging Face hoặc khóa/mật khẩu viết thẳng trong code."""
    findings = []
    for path in candidate_files(root):
        if not path.is_file():
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
