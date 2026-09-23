"""Configuration-driven SFT harness (full or LoRA) on transformers + trl + peft.

This is a scaffold: heavy dependencies import only inside `train`, and the run is skipped
(not failed) when a GPU or an optional library is missing. Nothing trains by default.
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

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FinetuneConfig":
        value = {key: item for key, item in value.items() if not key.startswith("_")}
        return cls(**{**value, "lora": LoraSettings(**value.get("lora", {}))})

    def validate(self) -> None:
        if self.method not in ("full", "lora"): raise ValueError(f"Unsupported finetune method: {self.method}; use 'full' or 'lora'")
        TrainingPlan("sft", self.dataset_path, self.base_model, self.seed, self.output_dir).validate()


def load_finetune_config(path: str | Path, **overrides: Any) -> FinetuneConfig:
    config = FinetuneConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    config = replace(config, **{key: value for key, value in overrides.items() if value is not None})
    config.validate(); return config


def resolve_base_model(config: FinetuneConfig) -> ModelConfig:
    return find_model_config(config.models_config, config.base_model)


def missing_requirements(config: FinetuneConfig) -> list[str]:
    """Names of absent optional libraries, plus 'cuda' when a GPU is required but unavailable."""
    packages = ["torch", "transformers", "trl", "datasets"] + (["peft"] if config.method == "lora" else [])
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
    if config.resume is False: return None
    if isinstance(config.resume, str): return config.resume
    checkpoint = latest_checkpoint(config.output_dir); return str(checkpoint) if checkpoint else None


def describe(config: FinetuneConfig) -> dict[str, Any]:
    model = resolve_base_model(config)
    return {"method": config.method, "base_model": {"name": model.name, "source": model.source, "revision": model.revision, "dtype": model.dtype}, "dataset_path": config.dataset_path, "output_dir": config.output_dir, "gradient_checkpointing": config.gradient_checkpointing, "resume_from": resume_target(config), "config": asdict(config)}


def train(config: FinetuneConfig) -> dict[str, Any]:
    config.validate(); model_config = resolve_base_model(config)
    missing = missing_requirements(config)
    if missing: return {"status": "skipped", "missing": missing}
    if model_config.dtype == "fp8": raise RuntimeError("FP8 training is unsupported until a runtime and hardware are tested")
    if not Path(config.dataset_path).exists(): raise FileNotFoundError(f"SFT dataset not found: {config.dataset_path}; run `python -m local_ai.data hf-sft` first")
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    seed_everything(config.seed)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(model_config.dtype)
    if dtype is None: raise ValueError(f"Unsupported dtype: {model_config.dtype}")
    tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer_source or model_config.source, revision=model_config.tokenizer_revision or model_config.revision, local_files_only=model_config.offline)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_config.source, revision=model_config.revision, torch_dtype=dtype, device_map=model_config.device_map, local_files_only=model_config.offline)
    model.config.use_cache = not config.gradient_checkpointing
    peft_config = None
    if config.method == "lora":
        from peft import LoraConfig
        peft_config = LoraConfig(r=config.lora.r, lora_alpha=config.lora.alpha, lora_dropout=config.lora.dropout, target_modules=config.lora.target_modules, task_type="CAUSAL_LM")
    dataset = load_dataset("json", data_files=config.dataset_path, split="train").select_columns(["messages"])
    args = SFTConfig(output_dir=config.output_dir, seed=config.seed, num_train_epochs=config.epochs, learning_rate=config.learning_rate, per_device_train_batch_size=config.per_device_batch_size, gradient_accumulation_steps=config.gradient_accumulation_steps, max_length=config.max_length, gradient_checkpointing=config.gradient_checkpointing, gradient_checkpointing_kwargs={"use_reentrant": False}, save_strategy="steps", save_steps=config.save_steps, save_total_limit=config.save_total_limit, logging_steps=config.logging_steps, bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16, report_to=[])
    tracker = RunTracker(config.output_dir, describe(config))
    trainer = SFTTrainer(model=model, args=args, train_dataset=dataset, processing_class=tokenizer, peft_config=peft_config)
    result = trainer.train(resume_from_checkpoint=resume_target(config))
    final = Path(config.output_dir) / ("adapter" if config.method == "lora" else "final")
    trainer.save_model(str(final)); tokenizer.save_pretrained(str(final))
    tracker.record_metrics({key: float(value) for key, value in result.metrics.items() if isinstance(value, (int, float))}); tracker.record_checkpoint(str(final))
    return {"status": "completed", "output": str(final), "metrics": result.metrics}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m local_ai.training.finetune")
    parser.add_argument("--config", required=True); parser.add_argument("--method", choices=("full", "lora")); parser.add_argument("--base-model"); parser.add_argument("--dataset-path"); parser.add_argument("--output-dir")
    parser.add_argument("--no-resume", action="store_true"); parser.add_argument("--dry-run", action="store_true", help="Print the resolved plan without importing training libraries")
    args = parser.parse_args(argv)
    config = load_finetune_config(args.config, method=args.method, base_model=args.base_model, dataset_path=args.dataset_path, output_dir=args.output_dir, resume=False if args.no_resume else None)
    print(json.dumps(describe(config) if args.dry_run else train(config), indent=2, default=str))


if __name__ == "__main__": main()
