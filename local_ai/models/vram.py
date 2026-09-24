"""Ước tính VRAM cho từng model trong danh sách model, chỉ dựa vào params_b (không tải model, không cần mạng).

Đây là ước lượng thô để chọn GPU, sai số có thể tới ±20%:
- trọng số = số tham số × số bit mỗi tham số ÷ 8. Nén 8bit/4bit tính thêm phần hằng số nén và các lớp
  giữ nguyên độ chính xác (embedding, lm_head, phần xử lý ảnh), nên lấy 8,5 và 4,5 bit thay vì 8 và 4;
- train LoRA/QLoRA = trọng số × 1,25 (activation khi bật gradient checkpointing, tham số LoRA và trạng thái
  optimizer, batch 1, độ dài khoảng 2048 token) + 1 GB cho CUDA context.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from local_ai.config.settings import load_model_configs
from local_ai.models.adapters import ModelConfig

BITS_PER_PARAM = {"bf16": 16.0, "8bit": 8.5, "4bit": 4.5}
TRAINING_FACTOR = 1.25
CUDA_CONTEXT_GB = 1.0


def weights_gb(params_b: float, precision: str) -> float:
    """GB (10^9 byte) cho riêng trọng số ở độ chính xác bf16, 8bit hoặc 4bit."""
    return params_b * BITS_PER_PARAM[precision] / 8


def training_gb(params_b: float, precision: str) -> float:
    """GB khi train LoRA (precision bf16) hoặc QLoRA (precision 4bit)."""
    return weights_gb(params_b, precision) * TRAINING_FACTOR + CUDA_CONTEXT_GB


def estimate(model: ModelConfig) -> dict[str, float | None]:
    if model.params_b is None: return {key: None for key in ("bf16", "8bit", "4bit", "lora_bf16", "qlora_4bit")}
    return {**{precision: weights_gb(model.params_b, precision) for precision in BITS_PER_PARAM}, "lora_bf16": training_gb(model.params_b, "bf16"), "qlora_4bit": training_gb(model.params_b, "4bit")}


def format_table(models: list[ModelConfig]) -> str:
    def cell(value: float | None) -> str: return "?" if value is None else f"{value:.1f}"
    lines = ["VRAM ước tính (GB) — chỉ là ước lượng từ params_b, chưa đo trên GPU thật.", "",
             "| Model | kind | Tham số (tỷ) | Trọng số bf16 | Trọng số 8bit | Trọng số 4bit | Train LoRA (bf16) | Train QLoRA (4bit) |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for model in models:
        values = estimate(model)
        lines.append(f"| {model.name} | {model.kind} | {'?' if model.params_b is None else f'{model.params_b:g}'} | " + " | ".join(cell(values[key]) for key in ("bf16", "8bit", "4bit", "lora_bf16", "qlora_4bit")) + " |")
    if any(model.params_b is None for model in models): lines += ["", "Dấu ? là model chưa khai báo params_b trong danh sách model."]
    lines += ["", f"Cách tính: trọng số = tỷ tham số × bit/tham số ÷ 8 (bf16 = 16, 8bit = 8,5, 4bit = 4,5 bit); train = trọng số × {TRAINING_FACTOR} + {CUDA_CONTEXT_GB:.0f} GB (batch 1, khoảng 2048 token, bật gradient checkpointing). Suy luận cần thêm bộ nhớ cho KV cache, tăng theo độ dài ngữ cảnh."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m local_ai.models.vram", description="In bảng VRAM ước tính cho mọi model trong danh sách model (không tải model).")
    parser.add_argument("--models", default=str(Path("configs") / "models" / "platform.json"), help="File danh sách model (mặc định configs/models/platform.json)")
    print(format_table(load_model_configs(parser.parse_args(argv).models)))


if __name__ == "__main__": main()
