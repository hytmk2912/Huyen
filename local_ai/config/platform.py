from __future__ import annotations
import json
from pathlib import Path
from local_ai.models.adapters import ModelConfig

def load_model_configs(path: str | Path) -> list[ModelConfig]:
    return [ModelConfig.from_dict(item) for item in json.loads(Path(path).read_text(encoding="utf-8"))["models"]]
