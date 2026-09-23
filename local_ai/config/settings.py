from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from local_ai.models.adapters import ModelConfig


@dataclass(frozen=True)
class RunSettings:
    seed: int
    max_iterations: int
    models: tuple[ModelConfig, ...]


def load_model_configs(path: str | Path) -> list[ModelConfig]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [ModelConfig.from_dict(item) for item in raw["models"]]


def load_settings(path: str | Path) -> RunSettings:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return RunSettings(seed=raw["seed"], max_iterations=raw["max_iterations"], models=tuple(load_model_configs(path)))
