"""Xuất model trong cấu hình ra file GGUF (mặc định f32) bằng script convert_hf_to_gguf.py của llama.cpp.

Trên Hugging Face chưa có sẵn bản GGUF f32 cho các model Huihui Qwen3, nên file được tạo cục bộ
từ trọng số gốc. Thư viện và llama.cpp chỉ cần khi chạy thật; `--dry-run` chỉ in lệnh sẽ chạy.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from local_ai.config.settings import find_model_config
from local_ai.models.adapters import ModelConfig

GGUF_OUTTYPES = {"f32", "f16", "bf16", "q8_0"}


def gguf_target(config: ModelConfig) -> tuple[Path, str]:
    if not config.gguf_file: raise ValueError(f"Model '{config.name}' chưa khai báo gguf_file trong cấu hình")
    outtype = config.gguf_outtype or "f32"
    if outtype not in GGUF_OUTTYPES: raise ValueError(f"Không hỗ trợ gguf_outtype: {outtype}; chọn một trong {', '.join(sorted(GGUF_OUTTYPES))}")
    return Path(config.gguf_file), outtype


def export_command(config: ModelConfig, snapshot_dir: str | Path, llama_cpp_dir: str | Path) -> list[str]:
    output, outtype = gguf_target(config)
    return [sys.executable, str(Path(llama_cpp_dir) / "convert_hf_to_gguf.py"), str(snapshot_dir), "--outtype", outtype, "--outfile", str(output)]


def export_gguf(config: ModelConfig, llama_cpp_dir: str | Path, dry_run: bool = False) -> dict[str, object]:
    output, outtype = gguf_target(config)
    script = Path(llama_cpp_dir) / "convert_hf_to_gguf.py"
    if dry_run: return {"model": config.name, "source": config.source, "outtype": outtype, "output": str(output), "command": export_command(config, "<thư mục snapshot>", llama_cpp_dir)}
    if output.exists(): return {"model": config.name, "output": str(output), "status": "exists"}
    if not script.exists(): raise FileNotFoundError(f"Không tìm thấy {script}; hãy clone llama.cpp và chỉ --llama-cpp tới thư mục đó")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error: raise RuntimeError("Hãy cài huggingface_hub để tải trọng số gốc trước khi xuất GGUF") from error
    snapshot = snapshot_download(config.source, revision=config.revision, token=os.environ.get("HF_TOKEN") or None, local_files_only=config.offline)
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(export_command(config, snapshot, llama_cpp_dir), check=True)
    return {"model": config.name, "output": str(output), "status": "exported"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m local_ai.models.gguf", description="Xuất model ra file GGUF")
    parser.add_argument("--model", required=True, help="Tên model trong danh sách model")
    parser.add_argument("--models-config", default="configs/models/platform.json", help="File danh sách model")
    parser.add_argument("--llama-cpp", default="llama.cpp", help="Thư mục mã nguồn llama.cpp")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ in lệnh sẽ chạy, không tải hay xuất gì")
    args = parser.parse_args(argv)
    print(json.dumps(export_gguf(find_model_config(args.models_config, args.model), args.llama_cpp, args.dry_run), indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
