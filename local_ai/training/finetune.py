"""Khung huấn luyện SFT (full, LoRA hoặc QLoRA) chạy theo cấu hình, dùng transformers + trl + peft.

QLoRA = `method: "lora"` trên model gốc nạp nén 4bit (`quantization: "4bit"` trong cấu hình
huấn luyện, hoặc khai báo sẵn trong danh sách model) bằng bitsandbytes.

Đây mới là khung: các thư viện nặng chỉ được import bên trong `train`, và lượt chạy sẽ bị
bỏ qua (không báo lỗi) khi thiếu GPU hoặc thư viện tùy chọn. Mặc định không huấn luyện gì.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from local_ai.config.settings import find_model_config
from local_ai.experiments.tracking import RunTracker, seed_everything
from local_ai.models.adapters import QUANTIZATIONS, TORCH_DTYPES, ModelConfig, model_loader_class, quantization_config, resolve_torch_dtype
from local_ai.training.hub import check_repo_id, download_last_checkpoint, downloaded_checkpoint
from local_ai.training.plans import TrainingPlan

FinetuneMethod = Literal["full", "lora"]


@dataclass(frozen=True)
class LoraSettings:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str | list[str] = "all-linear"


@dataclass(frozen=True)
class FinetuneConfig:
    base_model: str
    dataset_path: str
    output_dir: str
    models_config: str = "configs/models/platform.json"
    method: FinetuneMethod = "lora"
    lora: LoraSettings = field(default_factory=LoraSettings)
    seed: int = 0
    epochs: float = 1.0
    learning_rate: float = 2e-4
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_length: int = 2048
    gradient_checkpointing: bool = True
    save_steps: int = 200
    save_total_limit: int = 3
    logging_steps: int = 10
    resume: bool | str = True
    require_gpu: bool = True
    max_steps: int = -1  # > 0 thì dừng sau đúng số bước này (ví dụ chạy thử 2 bước), bỏ qua epochs
    quantization: str | None = None  # None: theo danh sách model; "4bit" | "8bit": nén khi nạp (QLoRA); "none": không nén
    dtype: str | None = None  # ghi đè dtype của model gốc, ví dụ "float16" cho GPU T4 trên Colab
    push_to_hub: bool = False  # đẩy checkpoint lên Hugging Face Hub trong lúc train (hub_strategy "checkpoint") để chạy lại thì train tiếp
    hub_model_id: str | None = None  # repo nhận checkpoint, dạng tên-người-dùng/tên-repo; token đọc từ biến môi trường HF_TOKEN
    hub_private: bool = True

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FinetuneConfig":
        value = {key: item for key, item in value.items() if not key.startswith("_")}
        return cls(**{**value, "lora": LoraSettings(**value.get("lora", {}))})

    def validate(self) -> None:
        if self.method not in ("full", "lora"): raise ValueError(f"Không hỗ trợ method finetune: {self.method}; hãy dùng 'full' hoặc 'lora'")
        if self.quantization not in (None, "none", *QUANTIZATIONS): raise ValueError(f"quantization phải là 4bit, 8bit hoặc none, không phải '{self.quantization}'")
        if self.dtype is not None and self.dtype not in TORCH_DTYPES: raise ValueError(f"dtype phải là một trong {', '.join(TORCH_DTYPES)}, không phải '{self.dtype}'")
        if self.push_to_hub: check_repo_id(self.hub_model_id)
        TrainingPlan("sft", self.dataset_path, self.base_model, self.seed, self.output_dir).validate()


def load_finetune_config(path: str | Path, **overrides: Any) -> FinetuneConfig:
    config = FinetuneConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    config = replace(config, **{key: value for key, value in overrides.items() if value is not None})
    config.validate(); return config


def resolve_base_model(config: FinetuneConfig) -> ModelConfig:
    model = find_model_config(config.models_config, config.base_model)
    if model.backend != "transformers": raise ValueError(f"Model '{model.name}' chạy qua server (backend {model.backend}), không fine-tune được; hãy chọn model có backend transformers")
    return model


def effective_quantization(config: FinetuneConfig, model: ModelConfig) -> str | None:
    """Kiểu nén khi nạp model gốc: cấu hình huấn luyện ghi đè danh sách model; full fine-tune không chạy trên model nén."""
    quantization = model.quantization if config.quantization is None else config.quantization
    quantization = None if quantization == "none" else quantization
    if quantization and config.method == "full": raise ValueError(f"Không full fine-tune được model nén {quantization}; hãy dùng method 'lora' (QLoRA) hoặc quantization 'none'")
    return quantization


def missing_requirements(config: FinetuneConfig) -> list[str]:
    """Tên các thư viện tùy chọn còn thiếu, thêm 'cuda' nếu cần GPU mà không có."""
    packages = ["torch", "transformers", "trl", "datasets"] + (["peft"] if config.method == "lora" else [])
    if effective_quantization(config, resolve_base_model(config)): packages.append("bitsandbytes")
    missing = [name for name in packages if importlib.util.find_spec(name) is None]
    if config.require_gpu and "torch" not in missing:
        import torch
        if not torch.cuda.is_available(): missing.append("cuda")
    return missing


def latest_checkpoint(output_dir: str | Path) -> Path | None:
    root = Path(output_dir)
    if not root.is_dir(): return None
    checkpoints = [(int(match.group(1)), path) for path in root.iterdir() if path.is_dir() and (match := re.fullmatch(r"checkpoint-(\d+)", path.name))]
    return max(checkpoints)[1] if checkpoints else None


def resume_target(config: FinetuneConfig) -> str | None:
    """Checkpoint để chạy tiếp: checkpoint-N mới nhất trên máy, nếu không có thì `last-checkpoint` đã tải từ Hub (thư mục `_hub/`)."""
    if config.resume is False: return None
    if isinstance(config.resume, str): return config.resume
    checkpoint = latest_checkpoint(config.output_dir) or downloaded_checkpoint(config.output_dir)
    return str(checkpoint) if checkpoint else None


def describe(config: FinetuneConfig) -> dict[str, Any]:
    model = resolve_base_model(config); quantization = effective_quantization(config, model)
    hub = {"push_to_hub": config.push_to_hub, "hub_model_id": config.hub_model_id, "private": config.hub_private, "hub_strategy": "checkpoint", "resume_from_hub": config.push_to_hub and config.resume is True} if config.push_to_hub else {"push_to_hub": False}
    return {"method": config.method, "quantization": quantization, "qlora": config.method == "lora" and quantization is not None, "base_model": {"name": model.name, "source": model.source, "revision": model.revision, "kind": model.kind, "params_b": model.params_b, "dtype": config.dtype or model.dtype}, "hub": hub, "dataset_path": config.dataset_path, "output_dir": config.output_dir, "gradient_checkpointing": config.gradient_checkpointing, "resume_from": resume_target(config), "config": asdict(config)}


def hub_arguments(config: FinetuneConfig) -> dict[str, Any]:
    """Tham số đẩy checkpoint lên Hub cho SFTConfig; token do thư viện tự đọc từ biến môi trường HF_TOKEN."""
    if not config.push_to_hub: return {}
    return {"push_to_hub": True, "hub_model_id": config.hub_model_id, "hub_strategy": "checkpoint", "hub_private_repo": config.hub_private}


MEASUREMENTS = "measurements.json"  # số đo thật của lần train gần nhất, trong output_dir (M15)


def checkpoint_step(path: str | None) -> int:
    """Số bước đã train của checkpoint sắp chạy tiếp; train từ đầu thì là 0."""
    state = Path(path) / "trainer_state.json" if path else None
    return int(json.loads(state.read_text(encoding="utf-8")).get("global_step", 0)) if state and state.is_file() else 0


def _cuda(torch: Any, name: str, *args: Any) -> Any:
    """Gọi hàm đo của torch.cuda nếu có. Phần đo lỗi thì bỏ trống, không làm hỏng lượt train."""
    function = getattr(torch.cuda, name, None)
    try:
        return function(*args) if function and torch.cuda.is_available() else None
    except Exception:
        return None


def training_measurements(torch: Any, config: FinetuneConfig, model: ModelConfig, trainer: Any, result: Any, start_step: int) -> dict[str, Any]:
    """Số đo thật của lần chạy này để so với ước tính (`python -m local_ai.training.calibrate`): GPU, VRAM đỉnh, thời gian, bước, token."""
    history = getattr(getattr(trainer, "state", None), "log_history", None) or []
    # TRL ghi số token thật (không tính phần đệm), đếm lại từ 0 khi chạy tiếp từ checkpoint; log cũ nạp từ checkpoint thì bỏ qua.
    tokens = [entry["num_tokens"] for entry in history if isinstance(entry.get("num_tokens"), (int, float)) and entry.get("step", start_step + 1) > start_step]
    gigabytes = lambda value: round(value / 1e9, 2) if isinstance(value, (int, float)) else None
    runtime = result.metrics.get("train_runtime") if isinstance(getattr(result, "metrics", None), dict) else None
    return {"model": model.name, "source": model.source, "params_b": model.params_b, "gpu": _cuda(torch, "get_device_name", 0) or "cpu",
            "quantization": effective_quantization(config, model), "dtype": config.dtype or model.dtype, "per_device_batch_size": config.per_device_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps, "max_length": config.max_length, "gradient_checkpointing": config.gradient_checkpointing,
            "start_step": start_step, "end_step": getattr(result, "global_step", None), "train_seconds": runtime, "num_tokens": max(tokens) if tokens else None,
            "peak_vram_gb": gigabytes(_cuda(torch, "max_memory_allocated")), "peak_reserved_gb": gigabytes(_cuda(torch, "max_memory_reserved")),
            "measured_at": datetime.now(timezone.utc).isoformat()}


def train(config: FinetuneConfig) -> dict[str, Any]:
    config.validate(); model_config = resolve_base_model(config); quantization = effective_quantization(config, model_config)
    missing = missing_requirements(config)
    if missing: return {"status": "skipped", "missing": missing}
    if model_config.dtype == "fp8": raise RuntimeError("Chưa hỗ trợ huấn luyện FP8 cho đến khi đã thử nghiệm runtime và phần cứng")
    if not Path(config.dataset_path).exists(): raise FileNotFoundError(f"Không tìm thấy dataset SFT: {config.dataset_path}; hãy chạy `python -m local_ai.data hf-sft` trước")
    import torch
    import transformers
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    seed_everything(config.seed)
    dtype = resolve_torch_dtype(torch, config.dtype or model_config.dtype)
    # sft.jsonl chỉ có chữ, nên kể cả model multimodal cũng dùng tokenizer (không cần phần xử lý ảnh).
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_config.tokenizer_source or model_config.source, revision=model_config.tokenizer_revision or model_config.revision, local_files_only=model_config.offline)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {"revision": model_config.revision, "torch_dtype": dtype, "device_map": model_config.device_map, "local_files_only": model_config.offline}
    if quantization: model_kwargs["quantization_config"] = quantization_config(transformers, quantization, dtype)
    model = model_loader_class(transformers, model_config.kind).from_pretrained(model_config.source, **model_kwargs)
    model.config.use_cache = not config.gradient_checkpointing
    if quantization:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=config.gradient_checkpointing, gradient_checkpointing_kwargs={"use_reentrant": False})
    peft_config = None
    if config.method == "lora":
        from peft import LoraConfig
        peft_config = LoraConfig(r=config.lora.r, lora_alpha=config.lora.alpha, lora_dropout=config.lora.dropout, target_modules=config.lora.target_modules, task_type="CAUSAL_LM")
    dataset = load_dataset("json", data_files=config.dataset_path, split="train").select_columns(["messages"])
    args = SFTConfig(output_dir=config.output_dir, seed=config.seed, num_train_epochs=config.epochs, max_steps=config.max_steps, learning_rate=config.learning_rate, per_device_train_batch_size=config.per_device_batch_size, gradient_accumulation_steps=config.gradient_accumulation_steps, max_length=config.max_length, gradient_checkpointing=config.gradient_checkpointing, gradient_checkpointing_kwargs={"use_reentrant": False}, save_strategy="steps", save_steps=config.save_steps, save_total_limit=config.save_total_limit, logging_steps=config.logging_steps, bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16, report_to=[], **hub_arguments(config))
    tracker = RunTracker(config.output_dir, describe(config))
    trainer = SFTTrainer(model=model, args=args, train_dataset=dataset, processing_class=tokenizer, peft_config=peft_config)
    resume = resume_target(config)
    if resume is None and config.push_to_hub and config.resume is True:  # Colab bị ngắt thì máy mới không còn checkpoint: lấy từ Hub
        downloaded = download_last_checkpoint(config.hub_model_id, config.output_dir); resume = str(downloaded) if downloaded else None
    result = trainer.train(resume_from_checkpoint=resume)
    measurements = training_measurements(torch, config, model_config, trainer, result, checkpoint_step(resume))
    # Ghi trước save_model: khi bật --push-to-hub, file này được đẩy lên repo cùng adapter.
    (Path(config.output_dir) / MEASUREMENTS).write_text(json.dumps(measurements, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    final = Path(config.output_dir) / ("adapter" if config.method == "lora" else "final")
    trainer.save_model(str(final)); tokenizer.save_pretrained(str(final))
    tracker.record_metrics({key: float(value) for key, value in result.metrics.items() if isinstance(value, (int, float))}); tracker.record_checkpoint(str(final))
    return {"status": "completed", "output": str(final), "steps": getattr(result, "global_step", None), "metrics": result.metrics, "measurements": measurements}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m local_ai.training.finetune")
    parser.add_argument("--config", required=True, help="File cấu hình huấn luyện (JSON)"); parser.add_argument("--method", choices=("full", "lora"), help="Ghi đè cách huấn luyện: full hoặc lora"); parser.add_argument("--base-model", help="Ghi đè tên model gốc trong danh sách model"); parser.add_argument("--dataset-path", help="Ghi đè đường dẫn sft.jsonl"); parser.add_argument("--output-dir", help="Ghi đè thư mục đầu ra")
    parser.add_argument("--quantization", choices=("4bit", "8bit", "none"), help="Ghi đè kiểu nén khi nạp model gốc: 4bit (QLoRA), 8bit hoặc none")
    parser.add_argument("--dtype", choices=TORCH_DTYPES, help="Ghi đè dtype của model gốc, ví dụ float16 cho GPU T4")
    parser.add_argument("--push-to-hub", action="store_true", help="Đẩy checkpoint lên Hugging Face Hub trong lúc train; chạy lại thì tự tải last-checkpoint về để train tiếp (token lấy từ biến môi trường HF_TOKEN)")
    parser.add_argument("--hub-model-id", help="Repo nhận checkpoint, dạng tên-người-dùng/tên-repo (mặc định riêng tư)")
    parser.add_argument("--no-resume", action="store_true", help="Không chạy tiếp từ checkpoint cũ, bắt đầu lại từ đầu"); parser.add_argument("--dry-run", action="store_true", help="Chỉ in kế hoạch đã phân giải, không import thư viện huấn luyện")
    args = parser.parse_args(argv)
    config = load_finetune_config(args.config, method=args.method, base_model=args.base_model, dataset_path=args.dataset_path, output_dir=args.output_dir, quantization=args.quantization, dtype=args.dtype, push_to_hub=True if args.push_to_hub else None, hub_model_id=args.hub_model_id, resume=False if args.no_resume else None)
    print(json.dumps(describe(config) if args.dry_run else train(config), indent=2, default=str))


if __name__ == "__main__": main()
