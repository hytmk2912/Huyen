"""Khung huấn luyện SFT (full hoặc LoRA) chạy theo cấu hình, dùng transformers + trl + peft.

Đây mới là khung: các thư viện nặng chỉ được import bên trong `train`, và lượt chạy sẽ bị
bỏ qua (không báo lỗi) khi thiếu GPU hoặc thư viện tùy chọn. Mặc định không huấn luyện gì.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from local_ai.config.settings import find_model_config
from local_ai.experiments.tracking import RunTracker, seed_everything
from local_ai.models.adapters import ModelConfig
from local_ai.training.plans import TrainingPlan

FinetuneMethod = Literal["full", "lora"]
DistributedStrategy = Literal["none", "ddp", "fsdp"]


@dataclass(frozen=True)
class LoraSettings:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str | list[str] = "all-linear"


@dataclass(frozen=True)
class DistributedSettings:
    """Huấn luyện nhiều GPU: ddp (mỗi GPU một bản model) hoặc fsdp (chia model qua các GPU). Luôn dùng BF16."""
    strategy: DistributedStrategy = "none"
    num_processes: int = 1
    num_nodes: int = 1


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
    train_dtype: str | None = None
    distributed: DistributedSettings = field(default_factory=DistributedSettings)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FinetuneConfig":
        value = {key: item for key, item in value.items() if not key.startswith("_")}
        return cls(**{**value, "lora": LoraSettings(**value.get("lora", {})), "distributed": DistributedSettings(**value.get("distributed", {}))})

    def validate(self) -> None:
        if self.method not in ("full", "lora"): raise ValueError(f"Không hỗ trợ method finetune: {self.method}; hãy dùng 'full' hoặc 'lora'")
        TrainingPlan("sft", self.dataset_path, self.base_model, self.seed, self.output_dir).validate()
        if self.distributed.strategy not in ("none", "ddp", "fsdp"): raise ValueError(f"Không hỗ trợ strategy phân tán: {self.distributed.strategy}; dùng none, ddp hoặc fsdp")
        if self.distributed.num_processes < 1 or self.distributed.num_nodes < 1: raise ValueError("num_processes và num_nodes phải từ 1 trở lên")
        if self.train_dtype == "fp8": raise ValueError("Chưa hỗ trợ huấn luyện FP8 cho đến khi đã thử nghiệm runtime và phần cứng")
        if self.distributed.strategy != "none" and self.train_dtype not in (None, "bfloat16"): raise ValueError("Huấn luyện phân tán chỉ hỗ trợ train_dtype bfloat16")


def load_finetune_config(path: str | Path, **overrides: Any) -> FinetuneConfig:
    config = FinetuneConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    config = replace(config, **{key: value for key, value in overrides.items() if value is not None})
    config.validate(); return config


def resolve_base_model(config: FinetuneConfig) -> ModelConfig:
    return find_model_config(config.models_config, config.base_model)


def training_dtype(config: FinetuneConfig, model: ModelConfig) -> str:
    """Kiểu số khi huấn luyện: train_dtype nếu có; khi chạy phân tán luôn là bfloat16; còn lại theo dtype của model."""
    if config.distributed.strategy != "none": return "bfloat16"
    return config.train_dtype or model.dtype


def training_arguments(config: FinetuneConfig, dtype: str) -> dict[str, Any]:
    """Tham số cho SFTConfig, tách riêng để kiểm tra được mà không cần torch."""
    kwargs: dict[str, Any] = {"output_dir": config.output_dir, "seed": config.seed, "num_train_epochs": config.epochs, "learning_rate": config.learning_rate, "per_device_train_batch_size": config.per_device_batch_size, "gradient_accumulation_steps": config.gradient_accumulation_steps, "max_length": config.max_length, "gradient_checkpointing": config.gradient_checkpointing, "gradient_checkpointing_kwargs": {"use_reentrant": False}, "save_strategy": "steps", "save_steps": config.save_steps, "save_total_limit": config.save_total_limit, "logging_steps": config.logging_steps, "bf16": dtype == "bfloat16", "fp16": dtype == "float16", "report_to": []}
    if config.distributed.strategy == "ddp": kwargs["ddp_find_unused_parameters"] = False
    if config.distributed.strategy == "fsdp":
        kwargs["fsdp"] = "full_shard auto_wrap"
        kwargs["fsdp_config"] = {"auto_wrap_policy": "TRANSFORMER_BASED_WRAP", "use_orig_params": True, "state_dict_type": "SHARDED_STATE_DICT", "activation_checkpointing": config.gradient_checkpointing}
        kwargs["gradient_checkpointing"] = False  # FSDP tự bật activation checkpointing qua fsdp_config
    return kwargs


def launch_command(config: FinetuneConfig, config_path: str | Path) -> list[str]:
    """Lệnh torchrun để chạy trên nhiều GPU; với strategy none thì chạy trực tiếp bằng python."""
    target = ["-m", "local_ai.training.finetune", "--config", str(config_path)]
    if config.distributed.strategy == "none": return ["python", *target]
    return ["torchrun", f"--nproc_per_node={config.distributed.num_processes}", f"--nnodes={config.distributed.num_nodes}", *target]


def missing_requirements(config: FinetuneConfig) -> list[str]:
    """Tên các thư viện tùy chọn còn thiếu, thêm 'cuda' nếu cần GPU mà không có."""
    packages = ["torch", "transformers", "trl", "datasets"] + (["peft"] if config.method == "lora" else [])
    missing = [name for name in packages if importlib.util.find_spec(name) is None]
    if config.require_gpu and "torch" not in missing:
        import torch
        if not torch.cuda.is_available(): missing.append("cuda")
        elif config.distributed.strategy != "none" and torch.cuda.device_count() < config.distributed.num_processes: missing.append(f"gpus>={config.distributed.num_processes}")
    return missing


def latest_checkpoint(output_dir: str | Path) -> Path | None:
    root = Path(output_dir)
    if not root.is_dir(): return None
    checkpoints = [(int(match.group(1)), path) for path in root.iterdir() if path.is_dir() and (match := re.fullmatch(r"checkpoint-(\d+)", path.name))]
    return max(checkpoints)[1] if checkpoints else None


def resume_target(config: FinetuneConfig) -> str | None:
    if config.resume is False: return None
    if isinstance(config.resume, str): return config.resume
    checkpoint = latest_checkpoint(config.output_dir); return str(checkpoint) if checkpoint else None


def describe(config: FinetuneConfig) -> dict[str, Any]:
    model = resolve_base_model(config)
    return {"method": config.method, "base_model": {"name": model.name, "source": model.source, "revision": model.revision, "dtype": model.dtype}, "dataset_path": config.dataset_path, "output_dir": config.output_dir, "gradient_checkpointing": config.gradient_checkpointing, "train_dtype": training_dtype(config, model), "distributed": asdict(config.distributed), "resume_from": resume_target(config), "config": asdict(config)}


def train(config: FinetuneConfig) -> dict[str, Any]:
    config.validate(); model_config = resolve_base_model(config)
    missing = missing_requirements(config)
    if missing: return {"status": "skipped", "missing": missing}
    dtype_name = training_dtype(config, model_config)
    if dtype_name == "fp8": raise RuntimeError("Chưa hỗ trợ huấn luyện FP8 cho đến khi đã thử nghiệm runtime và phần cứng")
    if not Path(config.dataset_path).exists(): raise FileNotFoundError(f"Không tìm thấy dataset SFT: {config.dataset_path}; hãy chạy `python -m local_ai.data hf-sft` trước")
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    seed_everything(config.seed)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(dtype_name)
    if dtype is None: raise ValueError(f"Không hỗ trợ dtype: {dtype_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer_source or model_config.source, revision=model_config.tokenizer_revision or model_config.revision, local_files_only=model_config.offline)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_config.source, revision=model_config.revision, torch_dtype=dtype, device_map=None if config.distributed.strategy != "none" else model_config.device_map, local_files_only=model_config.offline)
    model.config.use_cache = not config.gradient_checkpointing
    peft_config = None
    if config.method == "lora":
        from peft import LoraConfig
        peft_config = LoraConfig(r=config.lora.r, lora_alpha=config.lora.alpha, lora_dropout=config.lora.dropout, target_modules=config.lora.target_modules, task_type="CAUSAL_LM")
    dataset = load_dataset("json", data_files=config.dataset_path, split="train").select_columns(["messages"])
    args = SFTConfig(**training_arguments(config, dtype_name))
    tracker = RunTracker(config.output_dir, describe(config))
    trainer = SFTTrainer(model=model, args=args, train_dataset=dataset, processing_class=tokenizer, peft_config=peft_config)
    result = trainer.train(resume_from_checkpoint=resume_target(config))
    final = Path(config.output_dir) / ("adapter" if config.method == "lora" else "final")
    trainer.save_model(str(final)); tokenizer.save_pretrained(str(final))
    tracker.record_metrics({key: float(value) for key, value in result.metrics.items() if isinstance(value, (int, float))}); tracker.record_checkpoint(str(final))
    return {"status": "completed", "output": str(final), "metrics": result.metrics}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m local_ai.training.finetune")
    parser.add_argument("--config", required=True, help="File cấu hình huấn luyện (JSON)"); parser.add_argument("--method", choices=("full", "lora"), help="Ghi đè cách huấn luyện: full hoặc lora"); parser.add_argument("--base-model", help="Ghi đè tên model gốc trong danh sách model"); parser.add_argument("--dataset-path", help="Ghi đè đường dẫn sft.jsonl"); parser.add_argument("--output-dir", help="Ghi đè thư mục đầu ra")
    parser.add_argument("--no-resume", action="store_true", help="Không chạy tiếp từ checkpoint cũ, bắt đầu lại từ đầu"); parser.add_argument("--dry-run", action="store_true", help="Chỉ in kế hoạch đã phân giải, không import thư viện huấn luyện"); parser.add_argument("--print-launch", action="store_true", help="In lệnh torchrun để chạy trên nhiều GPU theo cấu hình distributed")
    args = parser.parse_args(argv)
    config = load_finetune_config(args.config, method=args.method, base_model=args.base_model, dataset_path=args.dataset_path, output_dir=args.output_dir, resume=False if args.no_resume else None)
    if args.print_launch: print(" ".join(launch_command(config, args.config))); return
    print(json.dumps(describe(config) if args.dry_run else train(config), indent=2, default=str))


if __name__ == "__main__": main()
