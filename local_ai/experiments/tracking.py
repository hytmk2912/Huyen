from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def seed_everything(seed: int) -> None:
    random.seed(seed)


class RunTracker:
    def __init__(self, directory: str | Path, config: dict[str, Any]):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    def record_metrics(self, metrics: dict[str, float]) -> None:
        (self.directory / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    def record_checkpoint(self, path: str) -> None:
        (self.directory / "checkpoint.txt").write_text(path + "\n", encoding="utf-8")
