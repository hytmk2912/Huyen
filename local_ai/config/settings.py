from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSettings:
    name: str
    backend: str
    capabilities: tuple[str, ...]


@dataclass(frozen=True)
class RunSettings:
    seed: int
    max_iterations: int
    models: tuple[ModelSettings, ...]


def load_settings(path: str | Path) -> RunSettings:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    models = tuple(ModelSettings(**model) for model in raw["models"])
    return RunSettings(seed=raw["seed"], max_iterations=raw["max_iterations"], models=models)
