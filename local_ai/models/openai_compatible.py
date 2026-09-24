"""Adapter gọi server model chạy cục bộ theo chuẩn OpenAI (`POST {base_url}/chat/completions`).

Dùng được với Ollama (`http://localhost:11434/v1`), llama.cpp server (`http://localhost:8080/v1`) và vLLM
(`http://localhost:8000/v1`). Chỉ dùng thư viện chuẩn (urllib), không cần thư viện `openai`.

Để không vô tình gọi API trả phí, adapter chỉ cho phép địa chỉ ở máy này hoặc mạng nội bộ, không dùng proxy
và không đi theo chuyển hướng (redirect).
"""
from __future__ import annotations

import ipaddress
import json
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

from local_ai.contracts import Message
from local_ai.models.adapters import ModelConfig

SERVER_HINT = "Server đã chạy chưa? Ví dụ: `ollama serve`, `llama-server -hf Qwen/Qwen2.5-0.5B-Instruct-GGUF --port 8080` hoặc `vllm serve <model> --port 8000`."


class ModelServerError(RuntimeError):
    """Lỗi khi gọi server model: không kết nối được, hết thời gian chờ, mã lỗi HTTP hoặc trả lời sai chuẩn."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):  # không đi theo redirect sang máy khác
        return None


def check_local_url(base_url: str, resolve: Callable[..., list] = socket.getaddrinfo) -> None:
    """Chỉ cho phép máy này hoặc mạng nội bộ (loopback, 10.x, 172.16–31.x, 192.168.x, 100.64.x...); địa chỉ Internet bị từ chối."""
    url = urllib.parse.urlsplit(base_url)
    if url.scheme not in ("http", "https") or not url.hostname: raise ValueError(f"base_url phải có dạng http://máy:cổng/v1, không phải '{base_url}'")
    try:
        addresses = {ipaddress.ip_address(url.hostname)}
    except ValueError:
        try:
            addresses = {ipaddress.ip_address(info[4][0].split("%")[0]) for info in resolve(url.hostname, url.port or 80)}
        except OSError as error: raise ModelServerError(f"Không phân giải được tên máy '{url.hostname}' trong {base_url}: {error}") from error
    external = sorted(str(address) for address in addresses if address.is_global)
    if external: raise ValueError(f"Chỉ cho phép server model ở máy này hoặc mạng nội bộ; '{url.hostname}' là địa chỉ Internet ({', '.join(external)}). Repo không gọi API bên ngoài hay API trả phí.")


class OpenAICompatibleAdapter:
    """Gửi hội thoại tới server local và trả về `choices[0].message.content`."""

    def __init__(self, config: ModelConfig):
        if config.backend != "openai_compatible" or not config.base_url: raise ValueError(f"Model '{config.name}' không khai báo backend openai_compatible và base_url")
        self.config, self._checked = config, False
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    @property
    def name(self) -> str: return self.config.name
    @property
    def capabilities(self) -> set[str]: return {capability.value for capability in self.config.capabilities}
    @property
    def endpoint(self) -> str: return self.config.base_url.rstrip("/") + "/chat/completions"

    def generate(self, messages: list[Message]) -> str:
        if any(message.images for message in messages): raise ValueError(f"Adapter server local chưa hỗ trợ gửi ảnh (model '{self.name}'); hãy dùng backend transformers cho model multimodal")
        if not self._checked: check_local_url(self.config.base_url); self._checked = True
        payload = {"model": self.config.source, "messages": [{"role": message.role, "content": message.content} for message in messages], "stream": False}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        key = os.environ.get(self.config.api_key_env) if self.config.api_key_env else None
        if key: headers["Authorization"] = f"Bearer {key}"
        request = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        try:
            with self._opener.open(request, timeout=self.config.timeout_s) as response: body = response.read()
        except urllib.error.HTTPError as error: raise ModelServerError(self._http_error(error)) from error
        except urllib.error.URLError as error:
            if isinstance(error.reason, TimeoutError): raise ModelServerError(self._timeout_message()) from error
            raise ModelServerError(f"Không kết nối được server model tại {self.config.base_url} ({error.reason}). {SERVER_HINT}") from error
        except TimeoutError as error: raise ModelServerError(self._timeout_message()) from error
        except OSError as error: raise ModelServerError(f"Mất kết nối với server model tại {self.config.base_url} ({type(error).__name__}: {error}). {SERVER_HINT}") from error
        return self._content(body)

    def _timeout_message(self) -> str:
        return f"Server model tại {self.config.base_url} không trả lời trong {self.config.timeout_s:g} giây; hãy tăng timeout_s hoặc kiểm tra server có đang quá tải không."

    def _http_error(self, error: urllib.error.HTTPError) -> str:
        detail = error.read().decode("utf-8", errors="replace")[:300].strip()
        hint = {401: f" Server yêu cầu khóa: hãy đặt biến môi trường {self.config.api_key_env or '<tên biến>'} và khai báo api_key_env.", 404: f" Kiểm tra base_url (thường kết thúc bằng /v1) và tên model '{self.config.source}' trên server."}.get(error.code, "")
        return f"Server model tại {self.config.base_url} trả mã lỗi HTTP {error.code}: {detail or error.reason}.{hint}"

    def _content(self, body: bytes) -> str:
        try:
            content = json.loads(body)["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as error:
            raise ModelServerError(f"Server model tại {self.config.base_url} trả dữ liệu không đúng chuẩn OpenAI (thiếu choices[0].message.content): {body[:200]!r}") from error
        if not isinstance(content, str): raise ModelServerError(f"Server model tại {self.config.base_url} trả về content không phải chuỗi: {content!r}")
        return content
