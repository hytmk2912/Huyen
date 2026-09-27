"""Xuất model để chạy bằng Ollama (M18): gộp LoRA adapter vào model gốc, chuyển sang GGUF, viết Modelfile.

`python -m local_ai.training.export --model light-colab --output .runs/gguf/light-da-train --llama-cpp /content/llama.cpp`
1. Nạp model gốc **không nén** theo revision đã ghim (dù mục trong danh sách model khai báo 4bit, vì không gộp được LoRA vào
   trọng số đã nén), nạp adapter của mục đó rồi gộp (`merge_and_unload`), lưu safetensors + tokenizer (có chat template) vào
   `<output>/hf`. `--no-adapter`: bỏ qua adapter, giữ nguyên model gốc; dùng để xuất model gốc theo đúng cùng đường (cùng
   revision, dtype, cách lượng tử) với model đã train, nên so sánh hai model là công bằng.
2. Chuyển `<output>/hf` sang `<output>/model.gguf` bằng `convert_hf_to_gguf.py` của llama.cpp (bản ghim trong notebook),
   mặc định `q8_0` (8 bit, gần như không giảm chất lượng; Ollama không tự lượng tử hóa GGUF khi nhập).
3. Viết `<output>/Modelfile`: chat template, lời hệ thống và tham số của họ model đó trong thư viện Ollama
   (`configs/ollama/<model_type>.json`, chọn theo `model_type` trong config.json), để cách hỏi giống hệt model cùng họ.
Sau đó: `ollama create <tên> -f <output>/Modelfile`.

Thư viện nặng (torch, transformers, peft) chỉ được import khi chạy thật; `--dry-run` chỉ in kế hoạch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from local_ai.config.settings import find_model_config
from local_ai.models.adapters import TORCH_DTYPES, ModelConfig, resolve_torch_dtype

ROOT = Path(__file__).resolve().parents[2]
OLLAMA_CONFIGS = ROOT / "configs" / "ollama"
OUTTYPES = ("q8_0", "f16", "bf16", "f32")
CONVERTER = "convert_hf_to_gguf.py"
LLAMA_CPP_HINT = "Tải llama.cpp bản đã ghim trước, ví dụ: git clone --depth 1 --branch b11205 https://github.com/ggml-org/llama.cpp /content/llama.cpp (cần thêm gói sentencepiece)."


def export_model_config(name: str, models: str | Path, no_adapter: bool = False, dtype: str | None = None) -> ModelConfig:
    """Cấu hình model để xuất: chỉ model chữ chạy bằng transformers; có adapter thì thư mục adapter phải có sẵn (trừ khi --no-adapter)."""
    config = find_model_config(models, name)
    if config.backend != "transformers": raise ValueError(f"Model '{config.name}' chạy qua server ({config.backend}), không có trọng số để xuất; hãy chọn model backend transformers")
    if config.kind != "text": raise ValueError(f"Model '{config.name}' là model ảnh + chữ; lệnh xuất GGUF hiện chỉ hỗ trợ model chữ (smoke, light)")
    if dtype is not None and dtype not in TORCH_DTYPES: raise ValueError(f"dtype phải là một trong {', '.join(TORCH_DTYPES)}, không phải '{dtype}'")
    return replace(config, adapter_path=None if no_adapter else config.adapter_path, quantization=None, dtype=dtype or config.dtype)


def converter_path(llama_cpp: str | Path) -> Path:
    script = Path(llama_cpp) / CONVERTER
    if not script.is_file(): raise FileNotFoundError(f"Không thấy {script}. {LLAMA_CPP_HINT}")
    return script


def plan(config: ModelConfig, output: Path, llama_cpp: str | Path, outtype: str) -> dict[str, Any]:
    script = Path(llama_cpp) / CONVERTER
    return {"model": config.name, "source": config.source, "revision": config.revision, "dtype": config.dtype, "adapter_path": config.adapter_path,
            "merged": config.adapter_path is not None, "hf_dir": str(output / "hf"), "gguf": str(output / "model.gguf"), "outtype": outtype,
            "modelfile": str(output / "Modelfile"), "converter": str(script), "converter_found": script.is_file(),
            "ollama_template": "theo model_type trong config.json: " + ", ".join(sorted(path.stem for path in OLLAMA_CONFIGS.glob("*.json")))}


def save_hf(config: ModelConfig, target: Path) -> dict[str, Any]:
    """Nạp model gốc (không nén) + adapter nếu có, gộp rồi lưu safetensors và tokenizer vào `target`."""
    try:
        import torch
        import transformers
    except ImportError as error: raise RuntimeError("Xuất model cần torch và transformers; hãy cài theo README") from error
    dtype = resolve_torch_dtype(torch, config.dtype)
    kwargs = {"revision": config.revision, "dtype": dtype, "local_files_only": config.offline}
    if config.device_map is not None and torch.cuda.is_available(): kwargs["device_map"] = config.device_map  # gộp trên GPU cho nhanh; không có GPU thì trên CPU
    model = transformers.AutoModelForCausalLM.from_pretrained(config.source, **kwargs)
    adapter = None
    if config.adapter_path:
        if not (Path(config.adapter_path) / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Không thấy adapter trong {config.adapter_path} (thiếu adapter_config.json); hãy tải adapter về trước (python -m local_ai.training.hub pull-adapter)")
        try:
            from peft import PeftModel
        except ImportError as error: raise RuntimeError("Gộp adapter cần thư viện peft; hãy chạy `python -m pip install peft`") from error
        model = PeftModel.from_pretrained(model, config.adapter_path).merge_and_unload()
        adapter = {"path": config.adapter_path, "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(Path(config.adapter_path).glob("adapter_model.*"))}}
    if any("lora_" in name for name, _ in model.named_parameters()): raise RuntimeError("Model sau khi gộp vẫn còn lớp LoRA; không xuất tiếp")
    if target.exists(): shutil.rmtree(target)
    model.save_pretrained(target, safe_serialization=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.tokenizer_source or config.source, revision=config.tokenizer_revision or config.revision, local_files_only=config.offline)
    tokenizer.save_pretrained(target)
    return {"model_type": model.config.model_type, "adapter": adapter}


def convert(hf_dir: Path, gguf: Path, llama_cpp: str | Path, outtype: str) -> None:
    """Chạy convert_hf_to_gguf.py của llama.cpp; lỗi thì báo tiếng Việt kèm phần cuối log của llama.cpp."""
    command = [sys.executable, str(converter_path(llama_cpp)), str(hf_dir), "--outfile", str(gguf), "--outtype", outtype]
    print("Chạy:", " ".join(command), file=sys.stderr, flush=True)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not gguf.is_file():
        tail = "\n".join((result.stderr or result.stdout).strip().splitlines()[-15:])
        hint = " Thiếu gói sentencepiece: chạy python -m pip install sentencepiece." if "sentencepiece" in tail else ""
        raise RuntimeError(f"Chuyển sang GGUF thất bại (mã thoát {result.returncode}).{hint}\n{tail}")


def ollama_settings(model_type: str) -> dict[str, Any]:
    """Chat template, lời hệ thống, tham số Ollama của họ model; kiểm tra mã băm để chắc là chép nguyên văn từ thư viện Ollama."""
    path = OLLAMA_CONFIGS / f"{model_type}.json"
    if not path.is_file():
        raise ValueError(f"Chưa có chat template Ollama cho model_type '{model_type}' (cần file {path.relative_to(ROOT)}); có: {', '.join(sorted(item.stem for item in OLLAMA_CONFIGS.glob('*.json')))}")
    settings = json.loads(path.read_text(encoding="utf-8"))
    for key, digest_key in (("template", "template_sha256"), ("system", "system_sha256")):
        if settings.get(key) is not None and hashlib.sha256(settings[key].encode("utf-8")).hexdigest() != settings.get(digest_key):
            raise ValueError(f"{path.relative_to(ROOT)}: {key} khác bản gốc của Ollama (mã băm không khớp {digest_key}); đừng sửa tay file này")
    return settings


def modelfile(gguf: Path, settings: dict[str, Any]) -> str:
    """Nội dung Modelfile: FROM file GGUF, TEMPLATE, SYSTEM (nếu có), PARAMETER của họ model trong thư viện Ollama."""
    for key in ("template", "system"):
        if '"""' in (settings.get(key) or ""): raise ValueError(f"{key} không được chứa ba dấu nháy kép liền nhau")
    lines = [f"FROM {gguf.resolve()}", f'TEMPLATE """{settings["template"]}"""']
    if settings.get("system"): lines.append(f'SYSTEM """{settings["system"]}"""')
    for name, value in settings.get("parameters", []):
        lines.append(f"PARAMETER {name} {json.dumps(value) if isinstance(value, str) else value}")
    return "\n".join(lines) + "\n"


def export(config: ModelConfig, output: str | Path, llama_cpp: str | Path, outtype: str = "q8_0", keep_hf: bool = False) -> dict[str, Any]:
    """Chạy cả 3 bước; trả về và ghi `<output>/export_info.json` (nguồn gốc model, adapter, file GGUF, Modelfile)."""
    if outtype not in OUTTYPES: raise ValueError(f"outtype phải là một trong {', '.join(OUTTYPES)}")
    output = Path(output); converter_path(llama_cpp)  # thiếu llama.cpp thì báo ngay, trước khi nạp model
    hf_dir, gguf = output / "hf", output / "model.gguf"
    output.mkdir(parents=True, exist_ok=True)
    saved = save_hf(config, hf_dir)
    convert(hf_dir, gguf, llama_cpp, outtype)
    settings = ollama_settings(saved["model_type"])
    (output / "Modelfile").write_text(modelfile(gguf, settings), encoding="utf-8")
    if not keep_hf: shutil.rmtree(hf_dir)  # model gốc 4B ở dạng safetensors khoảng 8 GB; GGUF đã đủ để chạy
    info = {"model": config.name, "source": config.source, "revision": config.revision, "dtype": config.dtype, "merged": saved["adapter"] is not None,
            "adapter": saved["adapter"], "model_type": saved["model_type"], "gguf": str(gguf), "gguf_mb": round(gguf.stat().st_size / 1e6, 1), "outtype": outtype,
            "modelfile": str(output / "Modelfile"), "ollama_template_from": settings.get("ollama_source"), "hf_dir": str(hf_dir) if keep_hf else None,
            "exported_at": datetime.now(timezone.utc).isoformat()}
    (output / "export_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m local_ai.training.export", description="Gộp LoRA adapter vào model gốc, xuất GGUF và viết Modelfile để chạy bằng Ollama.")
    parser.add_argument("--model", required=True, help="Tên model trong danh sách model, ví dụ light-colab (model gốc + adapter vừa train)")
    parser.add_argument("--models", default=str(ROOT / "configs" / "models" / "platform.json"), help="File danh sách model")
    parser.add_argument("--output", required=True, help="Thư mục đầu ra: model.gguf, Modelfile, export_info.json")
    parser.add_argument("--llama-cpp", default="/content/llama.cpp", help="Thư mục llama.cpp đã tải (có convert_hf_to_gguf.py)")
    parser.add_argument("--outtype", choices=OUTTYPES, default="q8_0", help="Kiểu số trong file GGUF (mặc định q8_0)")
    parser.add_argument("--dtype", choices=TORCH_DTYPES, help="Ghi đè dtype khi nạp model gốc (mặc định theo danh sách model)")
    parser.add_argument("--no-adapter", action="store_true", help="Không gộp adapter: xuất model gốc theo cùng cách, để so sánh với model đã train")
    parser.add_argument("--keep-hf", action="store_true", help="Giữ thư mục safetensors trung gian (mặc định xóa sau khi có GGUF để đỡ tốn chỗ)")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ in kế hoạch, không nạp model")
    args = parser.parse_args(argv)
    config = export_model_config(args.model, args.models, no_adapter=args.no_adapter, dtype=args.dtype)
    if args.dry_run:
        print(json.dumps({"status": "dry-run", **plan(config, Path(args.output), args.llama_cpp, args.outtype)}, ensure_ascii=False, indent=2)); return 0
    print(json.dumps({"status": "completed", **export(config, args.output, args.llama_cpp, args.outtype, args.keep_hf)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
