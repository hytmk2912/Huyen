"""Khung huấn luyện SFT chạy theo cấu hình: full, LoRA hoặc QLoRA (4-bit), dùng transformers + trl + peft.

Các thư viện nặng chỉ được import bên trong `train`; khi thiếu GPU hoặc thư viện, lượt chạy trả về
"skipped" thay vì báo lỗi. Sau khi lưu model, bộ đánh giá theo nhóm chạy tự động (eval_cases).
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
from local_ai.models.adapters import ModelConfig, quantization_settings
from local_ai.training.plans import TrainingPlan

FinetuneMethod = Literal["full", "lora", "qlora"]
METHODS = ("full", "lora", "qlora")
# Byte cho mỗi tham số khi chứa trọng số: bf16/fp16 = 2, 4-bit NF4 ≈ 0,55 (kể cả hệ số lượng tử hóa).
BYTES_PER_PARAM = {"bfloat16": 2.0, "float16": 2.0, "float32": 4.0, "nf4": 0.55}


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
    eval_cases: str | None = "data/eval/vi_trading_eval.jsonl"
    eval_max_new_tokens: int = 128

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FinetuneConfig":
        value = {key: item for key, item in value.items() if not key.startswith("_")}
        return cls(**{**value, "lora": LoraSettings(**value.get("lora", {}))})

    def validate(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"Không hỗ trợ method finetune: {self.method}; hãy dùng {', '.join(METHODS)}")
        TrainingPlan("sft", self.dataset_path, self.base_model, self.seed, self.output_dir).validate()


def load_finetune_config(path: str | Path, **overrides: Any) -> FinetuneConfig:
    config = FinetuneConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    config = replace(config, **{key: value for key, value in overrides.items() if value is not None})
    config.validate()
    return config


def resolve_base_model(config: FinetuneConfig) -> ModelConfig:
    return find_model_config(config.models_config, config.base_model)


def weight_format(config: FinetuneConfig, model: ModelConfig) -> str:
    """Cách chứa trọng số model gốc khi train: nf4 với QLoRA, còn lại theo dtype của model."""
    return "nf4" if config.method == "qlora" else model.dtype


def estimate_weight_memory_gb(model: ModelConfig, fmt: str) -> float | None:
    """Ước lượng bộ nhớ GPU chỉ để chứa trọng số (chưa tính activation, optimizer, KV cache)."""
    if not model.params_billion or fmt not in BYTES_PER_PARAM:
        return None
    return round(model.params_billion * BYTES_PER_PARAM[fmt], 1)


def missing_requirements(config: FinetuneConfig) -> list[str]:
    """Tên các thư viện tùy chọn còn thiếu, thêm 'cuda' nếu cần GPU mà không có."""
    packages = ["torch", "transformers", "trl", "datasets"]
    if config.method in ("lora", "qlora"):
        packages.append("peft")
    if config.method == "qlora":
        packages.append("bitsandbytes")
    missing = [name for name in packages if importlib.util.find_spec(name) is None]
    if config.require_gpu and "torch" not in missing:
        import torch

        if not torch.cuda.is_available():
            missing.append("cuda")
    return missing


def latest_checkpoint(output_dir: str | Path) -> Path | None:
    root = Path(output_dir)
    if not root.is_dir():
        return None
    checkpoints = [
        (int(match.group(1)), path)
        for path in root.iterdir()
        if path.is_dir() and (match := re.fullmatch(r"checkpoint-(\d+)", path.name))
    ]
    return max(checkpoints)[1] if checkpoints else None


def resume_target(config: FinetuneConfig) -> str | None:
    if config.resume is False:
        return None
    if isinstance(config.resume, str):
        return config.resume
    checkpoint = latest_checkpoint(config.output_dir)
    return str(checkpoint) if checkpoint else None


def training_arguments(config: FinetuneConfig) -> dict[str, Any]:
    """Tham số cho SFTConfig, tách riêng để kiểm tra được mà không cần torch. Luôn tính toán bằng bf16."""
    return {
        "output_dir": config.output_dir,
        "seed": config.seed,
        "num_train_epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "per_device_train_batch_size": config.per_device_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "max_length": config.max_length,
        "gradient_checkpointing": config.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "save_strategy": "steps",
        "save_steps": config.save_steps,
        "save_total_limit": config.save_total_limit,
        "logging_steps": config.logging_steps,
        "bf16": True,
        "report_to": [],
    }


def describe(config: FinetuneConfig) -> dict[str, Any]:
    model = resolve_base_model(config)
    fmt = weight_format(config, model)
    return {
        "method": config.method,
        "base_model": {"name": model.name, "source": model.source, "revision": model.revision, "dtype": model.dtype},
        "weight_format": fmt,
        "estimated_weight_memory_gb": estimate_weight_memory_gb(model, fmt),
        "dataset_path": config.dataset_path,
        "output_dir": config.output_dir,
        "gradient_checkpointing": config.gradient_checkpointing,
        "resume_from": resume_target(config),
        "eval_cases": config.eval_cases,
        "config": asdict(config),
    }


def evaluate_checkpoint(cases_path: str | Path, generate: Any, output_dir: str | Path) -> dict[str, Any]:
    """Chạy bộ đánh giá theo nhóm cho model vừa huấn luyện và ghi eval_report.json cạnh checkpoint."""
    from local_ai.evaluation.suites import load_cases, run_suites

    report = {"cases_file": str(cases_path), **run_suites(load_cases(cases_path), generate)}
    target = Path(output_dir) / "eval_report.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def _generator(model: Any, tokenizer: Any, max_new_tokens: int) -> Any:
    def generate(prompt: str) -> str:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return tokenizer.decode(output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)

    return generate


def _load_model(config: FinetuneConfig, model_config: ModelConfig) -> Any:
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    kwargs: dict[str, Any] = {
        "revision": model_config.revision,
        "device_map": model_config.device_map,
        "local_files_only": model_config.offline,
    }
    if config.method == "qlora":
        settings = quantization_settings("nf4")
        settings["bnb_4bit_compute_dtype"] = torch.bfloat16
        kwargs["quantization_config"] = BitsAndBytesConfig(**settings)
    else:
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(model_config.dtype)
        if dtype is None:
            raise ValueError(f"Không hỗ trợ dtype: {model_config.dtype}")
        kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_config.source, **kwargs)
    if config.method == "qlora":
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=config.gradient_checkpointing)
    model.config.use_cache = not config.gradient_checkpointing
    return model


def train(config: FinetuneConfig) -> dict[str, Any]:
    config.validate()
    model_config = resolve_base_model(config)
    missing = missing_requirements(config)
    if missing:
        return {"status": "skipped", "missing": missing}
    if not Path(config.dataset_path).exists():
        raise FileNotFoundError(
            f"Không tìm thấy dataset SFT: {config.dataset_path}; hãy chạy `python -m local_ai.data hf-sft` trước"
        )
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    seed_everything(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.tokenizer_source or model_config.source,
        revision=model_config.tokenizer_revision or model_config.revision,
        local_files_only=model_config.offline,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _load_model(config, model_config)
    peft_config = None
    if config.method in ("lora", "qlora"):
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            target_modules=config.lora.target_modules,
            task_type="CAUSAL_LM",
        )
    dataset = load_dataset("json", data_files=config.dataset_path, split="train").select_columns(["messages"])
    tracker = RunTracker(config.output_dir, describe(config))
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(**training_arguments(config)),
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    result = trainer.train(resume_from_checkpoint=resume_target(config))
    final = Path(config.output_dir) / ("final" if config.method == "full" else "adapter")
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))
    metrics = {key: float(value) for key, value in result.metrics.items() if isinstance(value, (int, float))}
    evaluation = None
    if config.eval_cases:
        trainer.model.eval()
        trainer.model.config.use_cache = True
        generate = _generator(trainer.model, tokenizer, config.eval_max_new_tokens)
        evaluation = evaluate_checkpoint(config.eval_cases, generate, final)
        metrics.update({f"eval_{suite}_accuracy": values["accuracy"] for suite, values in evaluation["suites"].items()})
    tracker.record_metrics(metrics)
    tracker.record_checkpoint(str(final))
    return {"status": "completed", "output": str(final), "metrics": result.metrics, "evaluation": evaluation}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m local_ai.training.finetune")
    parser.add_argument("--config", required=True, help="File cấu hình huấn luyện (JSON)")
    parser.add_argument("--method", choices=METHODS, help="Ghi đè cách huấn luyện: full, lora hoặc qlora")
    parser.add_argument("--base-model", help="Ghi đè tên model gốc trong danh sách model")
    parser.add_argument("--dataset-path", help="Ghi đè đường dẫn sft.jsonl")
    parser.add_argument("--output-dir", help="Ghi đè thư mục đầu ra")
    parser.add_argument("--no-resume", action="store_true", help="Không chạy tiếp từ checkpoint cũ, bắt đầu lại từ đầu")
    parser.add_argument(
        "--dry-run", action="store_true", help="Chỉ in kế hoạch đã phân giải, không import thư viện huấn luyện"
    )
    args = parser.parse_args(argv)
    config = load_finetune_config(
        args.config,
        method=args.method,
        base_model=args.base_model,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        resume=False if args.no_resume else None,
    )
    print(json.dumps(describe(config) if args.dry_run else train(config), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
