"""Kiểm tra token cho runtime và gateway (từ `require_token` / `auth` của repo Agent)."""
from __future__ import annotations

import hmac


class AuthError(Exception):
    """Lỗi xác thực; `status` là mã HTTP tương ứng (503: chưa cấu hình token, 401: token sai)."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def require_token(expected: str | None, authorization: str | None) -> None:
    """Chấp nhận đúng header `Authorization: Bearer <token>`. So sánh bằng hmac.compare_digest để tránh đo thời gian."""
    if not expected: raise AuthError(503, "Chưa cấu hình token; hãy đặt biến môi trường chứa token trước khi bật dịch vụ")
    provided = authorization or ""
    if not hmac.compare_digest(provided.encode("utf-8"), f"Bearer {expected}".encode("utf-8")): raise AuthError(401, "Token không đúng")
