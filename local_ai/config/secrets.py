"""Nơi duy nhất đọc secret (token, khóa) của repo.

Mọi secret chỉ đọc từ biến môi trường (trên Colab/Kaggle: notebook lấy từ Secrets rồi đặt vào biến môi trường).
Không có giá trị mặc định: thiếu biến bắt buộc thì báo lỗi tiếng Việt và dừng (`MissingSecretError` là một `SystemExit`).
Không in, không ghi log giá trị secret; thông báo lỗi chỉ ghi tên biến. Dùng `redact()` trước khi in chuỗi có thể chứa secret.
"""
from __future__ import annotations

import os
from typing import Mapping

# Tên biến môi trường -> mô tả ngắn (đồng bộ với .env.example và SECURITY.md).
KNOWN_SECRETS = {
    "HF_TOKEN": "token Hugging Face: đọc dataset/model riêng tư, đẩy adapter và số đo lên repo riêng tư",
    "LOCAL_AI_GATEWAY_TOKEN": "token của gateway HTTP local (ít nhất 16 ký tự), chỉ cần khi bật gateway",
}
MASK = "***"
MIN_REDACT_LENGTH = 4  # chuỗi quá ngắn thì không thay, tránh che nhầm chữ thường


class MissingSecretError(SystemExit):
    """Thiếu biến môi trường chứa secret bắt buộc. Kế thừa SystemExit: lệnh dừng với thông báo rõ ràng, không in stack trace."""

    def __init__(self, name: str, reason: str = "") -> None:
        self.name = name
        detail = f" ({reason})" if reason else ""
        super().__init__(f"Thiếu biến môi trường {name}{detail}. Hãy đặt biến này (xem .env.example và SECURITY.md); "
                         "không ghi giá trị vào code, cấu hình hay notebook.")


def _environ(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def require_secret(name: str, environ: Mapping[str, str] | None = None, min_length: int = 1, reason: str = "") -> str:
    """Giá trị secret bắt buộc. Thiếu, rỗng hoặc ngắn hơn `min_length` thì dừng bằng MissingSecretError (không có giá trị dự phòng)."""
    value = (_environ(environ).get(name) or "").strip()
    if len(value) < max(1, min_length):
        raise MissingSecretError(name, reason or (f"cần ít nhất {min_length} ký tự" if min_length > 1 else KNOWN_SECRETS.get(name, "")))
    return value


def optional_secret(name: str, environ: Mapping[str, str] | None = None) -> str | None:
    """Secret không bắt buộc (ví dụ HF_TOKEN khi chỉ đọc dataset công khai): có thì trả giá trị, không có thì None.
    Không bao giờ trả một giá trị mặc định thay thế."""
    value = (_environ(environ).get(name) or "").strip()
    return value or None


def hf_token(required: bool = False, reason: str = "") -> str | None:
    """HF_TOKEN từ biến môi trường. `required=True` (đẩy lên Hub, repo riêng tư) thì thiếu là dừng."""
    return require_secret("HF_TOKEN", reason=reason) if required else optional_secret("HF_TOKEN")


def redact(text: str, environ: Mapping[str, str] | None = None, names: tuple[str, ...] | None = None) -> str:
    """Thay mọi giá trị secret đang có trong biến môi trường (KNOWN_SECRETS hoặc `names`) bằng *** trong `text`."""
    env = _environ(environ)
    for name in names or tuple(KNOWN_SECRETS):
        value = (env.get(name) or "").strip()
        if len(value) >= MIN_REDACT_LENGTH: text = text.replace(value, MASK)
    return text
