"""Cố định phiên bản (commit) và chạy thử model/tokenizer trong cấu hình.

- `pin`: hỏi Hugging Face commit hiện tại của `revision` (ví dụ "main") và ghi mã commit vào
  file danh sách model, để lần tải sau luôn ra đúng một phiên bản.
- `run`: nạp tokenizer + model, sinh một câu trả lời ngắn và ghi báo cáo. Thiếu GPU hoặc thư viện
  thì trả về "skipped" thay vì báo lỗi.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from local_ai.config.settings import find_model_config
from local_ai.contracts import Message
from local_ai.models.adapters import HuggingFaceModelAdapter, ModelConfig

COMMIT = re.compile(r"[0-9a-f]{40}")


def is_pinned(config: ModelConfig) -> bool:
    return bool(COMMIT.fullmatch(config.revision)) and bool(COMMIT.fullmatch(config.tokenizer_revision or config.revision))


def resolve_commit(repo: str, revision: str, api: Any = None) -> str:
    if COMMIT.fullmatch(revision): return revision
    if api is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as error: raise RuntimeError("Hãy cài huggingface_hub để tra mã commit trên Hugging Face") from error
        api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    return api.model_info(repo, revision=revision).sha


def pin_model(models_config: str | Path, name: str, api: Any = None) -> dict[str, str]:
    """Ghi mã commit cố định vào revision/tokenizer_revision của model `name` trong file cấu hình."""
    path = Path(models_config); raw = json.loads(path.read_text(encoding="utf-8"))
    entry = next((item for item in raw["models"] if item["name"] == name), None)
    if entry is None: raise LookupError(f"Model '{name}' chưa được khai báo trong {path}")
    entry["revision"] = resolve_commit(entry["source"], entry.get("revision", "main"), api)
    if entry.get("tokenizer_source") or entry.get("tokenizer_revision"):
        entry["tokenizer_revision"] = resolve_commit(entry.get("tokenizer_source") or entry["source"], entry.get("tokenizer_revision") or "main", api)
    path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"model": name, "revision": entry["revision"], "tokenizer_revision": entry.get("tokenizer_revision") or entry["revision"]}


def missing_requirements(require_gpu: bool = True) -> list[str]:
    missing = [name for name in ("torch", "transformers") if importlib.util.find_spec(name) is None]
    if require_gpu and "torch" not in missing:
        import torch
        if not torch.cuda.is_available(): missing.append("cuda")
    return missing


def smoke_test(config: ModelConfig, prompt: str = "Trả lời ngắn gọn: 2 + 3 bằng mấy?", require_gpu: bool = True, report_dir: str | Path = ".runs/smoke", adapter_factory=HuggingFaceModelAdapter) -> dict[str, Any]:
    missing = missing_requirements(require_gpu) if adapter_factory is HuggingFaceModelAdapter else []
    if missing: return {"model": config.name, "status": "skipped", "missing": missing, "pinned": is_pinned(config)}
    started = time.monotonic(); adapter = adapter_factory(config); adapter.load()
    reply = adapter.generate([Message("user", prompt)])
    report = {"model": config.name, "status": "passed" if reply.strip() else "failed", "pinned": is_pinned(config), "tokenizer": adapter.tokenizer_metadata(), "prompt": prompt, "reply": reply, "seconds": round(time.monotonic() - started, 2)}
    target = Path(report_dir) / f"{config.name}.json"; target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m local_ai.models.smoke", description="Cố định phiên bản và chạy thử model")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in (("pin", "Ghi mã commit cố định vào cấu hình"), ("run", "Nạp model và sinh thử một câu trả lời")):
        item = commands.add_parser(name, help=text)
        item.add_argument("--model", default="primary", help="Tên model trong danh sách model")
        item.add_argument("--models-config", default="configs/models/platform.json", help="File danh sách model")
    commands.choices["run"].add_argument("--allow-cpu", action="store_true", help="Cho phép chạy thử khi không có GPU")
    args = parser.parse_args(argv)
    if args.command == "pin": result = pin_model(args.models_config, args.model)
    else: result = smoke_test(find_model_config(args.models_config, args.model), require_gpu=not args.allow_cpu)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
