from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TrainingMethod = Literal["sft", "preference_optimization", "rft"]


@dataclass(frozen=True)
class TrainingPlan:
    method: TrainingMethod
    dataset_manifest: str
    base_model: str
    seed: int
    output_dir: str

    def validate(self) -> None:
        if not self.dataset_manifest or not self.base_model or not self.output_dir:
            raise ValueError("Kế hoạch huấn luyện cần có dataset, model gốc và thư mục đầu ra")
