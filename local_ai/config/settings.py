from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from local_ai.models.adapters import ModelConfig, UnsafeModelError, unpinned_message


@dataclass(frozen=True)
class RunSettings:
    seed: int
    max_iterations: int
    models: tuple[ModelConfig, ...]


def load_model_configs(path: str | Path) -> list[ModelConfig]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    configs = [ModelConfig.from_dict(item) for item in raw["models"]]
    names = [config.name for config in configs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates: raise ValueError(f"Tên model bị trùng trong {path}: {', '.join(duplicates)}")
    for config in configs:  # model tải từ Hugging Face phải ghim revision (mã commit 40 ký tự), cả model lẫn tokenizer
        if config.backend != "transformers": continue
        for source, revision, what in ((config.source, config.revision, "model"), (config.tokenizer_source or config.source, config.tokenizer_revision or (config.revision if config.tokenizer_source in (None, config.source) else None), "tokenizer")):
            message = unpinned_message(source, revision, f"{what} của mục '{config.name}':")
            if message: raise UnsafeModelError(f"{path}: {message}")
    return configs


def find_model_config(path: str | Path, name: str) -> ModelConfig:
    for config in load_model_configs(path):
        if config.name == name: return config
    raise LookupError(f"Model '{name}' chưa được khai báo trong {path}")


def load_settings(path: str | Path) -> RunSettings:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return RunSettings(seed=raw["seed"], max_iterations=raw["max_iterations"], models=tuple(load_model_configs(path)))
