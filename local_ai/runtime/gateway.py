"""Gateway HTTP hàng đợi job (từ `gateway.py` của repo Agent), viết bằng `http.server` của thư viện chuẩn.

An toàn mặc định: tắt (enabled: false), chỉ được nghe 127.0.0.1, bắt buộc token (`Authorization: Bearer <token>`,
token ít nhất 16 ký tự, đọc từ biến môi trường ghi ở `token_env`). Không có endpoint nào chạy lệnh qua mạng.

Endpoint: GET /health; POST /v1/jobs; GET /v1/jobs/<id>; POST /v1/jobs/<id>/claim; POST /v1/jobs/<id>/result.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from local_ai.runtime.auth import AuthError, require_token
from local_ai.runtime.jobs import JobError, JobQueue

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "runtime" / "gateway.json"
ALLOWED_HOST = "127.0.0.1"
MIN_TOKEN_LENGTH = 16
MAX_BODY = 64 * 1024
_JOB_PATH = re.compile(r"^/v1/jobs/([0-9a-f-]{36})(/claim|/result)?$")


@dataclass(frozen=True)
class GatewayConfig:
    enabled: bool = False
    host: str = ALLOWED_HOST
    port: int = 8765
    token_env: str = "LOCAL_AI_GATEWAY_TOKEN"
    max_jobs: int = 1000

    @classmethod
    def from_file(cls, path: str | Path = DEFAULT_CONFIG) -> "GatewayConfig":
        return cls(**{key: value for key, value in json.loads(Path(path).read_text(encoding="utf-8")).items() if not key.startswith("_")})


class GatewayRefused(RuntimeError):
    """Gateway không được bật vì cấu hình chưa an toàn (đang tắt, địa chỉ ngoài 127.0.0.1, thiếu token)."""


def create_server(config: GatewayConfig, environ: dict[str, str] | None = None) -> ThreadingHTTPServer:
    """Kiểm tra cấu hình rồi tạo server (chưa chạy). Dùng `server.serve_forever()` để chạy."""
    if not config.enabled: raise GatewayRefused("Gateway đang tắt; muốn bật phải đặt enabled: true trong cấu hình")
    if config.host != ALLOWED_HOST: raise GatewayRefused(f"Gateway chỉ được nghe {ALLOWED_HOST}, không nghe '{config.host}'")
    token = (environ if environ is not None else os.environ).get(config.token_env, "")
    if len(token) < MIN_TOKEN_LENGTH: raise GatewayRefused(f"Thiếu token: hãy đặt biến môi trường {config.token_env} (ít nhất {MIN_TOKEN_LENGTH} ký tự) trước khi bật gateway")
    queue = JobQueue(max_jobs=config.max_jobs)

    class Handler(BaseHTTPRequestHandler):
        server_version = "LocalAIGateway/0.1"

        def log_message(self, format: str, *args: Any) -> None:  # không in từng request ra màn hình
            pass

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def _body(self) -> Any:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY: raise ValueError(f"Nội dung quá lớn (tối đa {MAX_BODY} byte)")
            return json.loads(self.rfile.read(length) or b"{}")

        def _handle(self, method: str) -> None:
            try:
                if method == "GET" and self.path == "/health": return self._send(200, {"ok": True})
                require_token(token, self.headers.get("Authorization"))
                if method == "POST" and self.path == "/v1/jobs":
                    body = self._body(); job = queue.create(body.get("instruction") if isinstance(body, dict) else None)
                    return self._send(201, {"job_id": job.id, "status": job.status})
                match = _JOB_PATH.match(self.path)
                if not match: return self._send(404, {"error": "Không có đường dẫn này"})
                job_id, action = match.groups()
                if method == "GET" and action is None: return self._send(200, queue.get(job_id).as_dict())
                if method == "POST" and action == "/claim": job = queue.claim(job_id); return self._send(200, {"job_id": job.id, "instruction": job.instruction, "status": job.status})
                if method == "POST" and action == "/result": job = queue.submit_result(job_id, self._body()); return self._send(200, {"job_id": job.id, "status": job.status})
                return self._send(405, {"error": "Không hỗ trợ phương thức này"})
            except AuthError as error: self._send(error.status, {"error": str(error)})
            except JobError as error: self._send(error.status, {"error": str(error)})
            except (ValueError, json.JSONDecodeError) as error: self._send(400, {"error": f"Yêu cầu không hợp lệ: {error}"})

        def do_GET(self) -> None: self._handle("GET")
        def do_POST(self) -> None: self._handle("POST")

    return ThreadingHTTPServer((config.host, config.port), Handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m local_ai.runtime.gateway", description="Chạy gateway hàng đợi job (tắt mặc định, chỉ nghe 127.0.0.1, bắt buộc token).")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="File cấu hình gateway (mặc định configs/runtime/gateway.json, đang tắt)")
    config = GatewayConfig.from_file(parser.parse_args(argv).config)
    try:
        server = create_server(config)
    except GatewayRefused as error:
        print(f"Không bật gateway: {error}"); return 1
    print(f"Gateway đang nghe http://{config.host}:{server.server_address[1]} (Ctrl+C để dừng)")
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
