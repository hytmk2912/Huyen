from __future__ import annotations

import json
from pathlib import Path

from local_ai.models.adapters import ModelConfig


def load_model_configs(path: str | Path) -> list[ModelConfig]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    configs = [
        ModelConfig.from_dict({k: v for k, v in item.items() if not k.startswith("_")}) for item in raw["models"]
    ]
    names = [config.name for config in configs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Tên model bị trùng trong {path}: {', '.join(duplicates)}")
    return configs


def find_model_config(path: str | Path, name: str) -> ModelConfig:
    for config in load_model_configs(path):
        if config.name == name:
            return config
    raise LookupError(f"Model '{name}' chưa được khai báo trong {path}")
