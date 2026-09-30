"""Ước tính VRAM cho từng model trong danh sách model, chỉ dựa vào params_b (không tải model, không cần mạng).

Đây là ước lượng thô để chọn GPU, sai số có thể tới ±20%:
- trọng số = số tham số × số bit mỗi tham số ÷ 8. Nén 8bit/4bit tính thêm phần hằng số nén và các lớp
  giữ nguyên độ chính xác (embedding, lm_head, phần xử lý ảnh), nên lấy 8,5 và 4,5 bit thay vì 8 và 4;
- train LoRA/QLoRA = trọng số × 1,25 (activation khi bật gradient checkpointing, tham số LoRA và trạng thái
  optimizer, batch 1, độ dài khoảng 2048 token) + 1 GB cho CUDA context;
- model nén (QLoRA 4bit, 8bit) cộng thêm KBIT_OVERHEAD_GB × √(tỷ tham số). Khi train model nén, embedding bị đổi
  sang float32 và phần logits trên bộ từ vựng (khoảng 152 nghìn token với Qwen) rất lớn so với trọng số 4bit; phần này
  tăng chậm hơn số tham số nên tính theo căn bậc hai. Hằng số đo thật trên Colab T4 ngày 25/9/2026 (mốc M15):
  `light` (Qwen3-4B) dùng 8,17 GB, công thức cũ chỉ ước tính 3,8 GB; `smoke` (0,5B) dùng 2,5 GB, công thức mới
  ước tính 3,2 GB. Model 27B chưa đo: con số của nó là suy ra, cần đo lại khi làm M17.
- train LoRA bf16 (không nén) chưa đo thật, vẫn dùng công thức cũ.
- độ dài chuỗi (seq len, M25): công thức trên hiệu chỉnh ở REFERENCE_SEQ_LEN = 2048 token, batch 1. Phần thêm khi train
  (0,25 × trọng số và phần của model nén) có phần tăng theo số token (activation, logits) và phần không đổi (tham số
  LoRA, trạng thái optimizer, embedding float32). Chưa đo tách riêng hai phần, nên giả định một nửa (SEQ_SHARE) tăng tỉ
  lệ với số token, nửa còn lại không đổi. Ở 2048 token kết quả giữ nguyên như trước; ở độ dài khác chỉ là suy ra.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from local_ai.config.settings import load_model_configs
from local_ai.models.adapters import ModelConfig

ROOT = Path(__file__).resolve().parents[2]

BITS_PER_PARAM = {"bf16": 16.0, "8bit": 8.5, "4bit": 4.5}
TRAINING_FACTOR = 1.25
CUDA_CONTEXT_GB = 1.0
KBIT_OVERHEAD_GB = 2.67  # GB × √(tỷ tham số): (8,17 GB đo với light − 2,26 GB trọng số 4bit × 1,25) ÷ √4,02
REFERENCE_SEQ_LEN = 2048  # độ dài chuỗi (token, batch 1) mà các hằng số trên được hiệu chỉnh
SEQ_SHARE = 0.5  # phần của "phần thêm khi train" tăng tỉ lệ với số token; giả định, chưa đo (M25)


def weights_gb(params_b: float, precision: str) -> float:
    """GB (10^9 byte) cho riêng trọng số ở độ chính xác bf16, 8bit hoặc 4bit."""
    return params_b * BITS_PER_PARAM[precision] / 8


def kbit_overhead_gb(params_b: float, precision: str) -> float:
    """Phần thêm khi train model nén 4bit/8bit (embedding float32, logits); model không nén thì bằng 0."""
    return KBIT_OVERHEAD_GB * math.sqrt(params_b) if precision in ("4bit", "8bit") else 0.0


def training_gb(params_b: float, precision: str, seq_len: int = REFERENCE_SEQ_LEN) -> float:
    """GB khi train LoRA (precision bf16) hoặc QLoRA (precision 4bit), batch 1, mỗi dòng dài tối đa seq_len token."""
    if seq_len <= 0: raise ValueError(f"seq_len phải lớn hơn 0, nhận được {seq_len}")
    weights = weights_gb(params_b, precision)
    extra = weights * (TRAINING_FACTOR - 1) + kbit_overhead_gb(params_b, precision)  # phần thêm khi train, hiệu chỉnh ở REFERENCE_SEQ_LEN
    return weights + extra * (1 - SEQ_SHARE + SEQ_SHARE * seq_len / REFERENCE_SEQ_LEN) + CUDA_CONTEXT_GB


def max_seq_len(params_b: float, precision: str, budget_gb: float, step: int = 64) -> int:
    """Độ dài chuỗi lớn nhất (bội số của step) mà ước tính train còn vừa budget_gb; 0 nếu độ dài nào cũng không vừa."""
    weights = weights_gb(params_b, precision)
    extra = weights * (TRAINING_FACTOR - 1) + kbit_overhead_gb(params_b, precision)
    fixed = weights + extra * (1 - SEQ_SHARE) + CUDA_CONTEXT_GB  # phần không đổi theo độ dài
    per_token = extra * SEQ_SHARE / REFERENCE_SEQ_LEN
    length = max(0, int((budget_gb - fixed) / per_token) // step * step)
    while length > 0 and training_gb(params_b, precision, length) > budget_gb: length -= step  # phòng sai số làm tròn
    return length


def estimate(model: ModelConfig, seq_len: int = REFERENCE_SEQ_LEN) -> dict[str, float | None]:
    if model.params_b is None: return {key: None for key in ("bf16", "8bit", "4bit", "lora_bf16", "qlora_4bit")}
    return {**{precision: weights_gb(model.params_b, precision) for precision in BITS_PER_PARAM}, "lora_bf16": training_gb(model.params_b, "bf16", seq_len), "qlora_4bit": training_gb(model.params_b, "4bit", seq_len)}


def format_table(models: list[ModelConfig], seq_len: int = REFERENCE_SEQ_LEN, budget_gb: float | None = None) -> str:
    def cell(value: float | None) -> str: return "?" if value is None else f"{value:.1f}"
    budget = f" | Độ dài lớn nhất vừa {budget_gb:g} GB (QLoRA)" if budget_gb else ""
    lines = [f"VRAM ước tính (GB) — ước lượng từ params_b, khi train mỗi dòng dài tối đa {seq_len} token; phần QLoRA đã hiệu chỉnh theo số đo thật của model 4B trên T4.", "",
             "| Model | kind | Tham số (tỷ) | Trọng số bf16 | Trọng số 8bit | Trọng số 4bit | Train LoRA (bf16) | Train QLoRA (4bit)" + budget + " |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---:" + (" | ---:" if budget else "") + " |"]
    for model in models:
        values = estimate(model, seq_len)
        longest = "" if not budget else " | " + ("?" if model.params_b is None else str(max_seq_len(model.params_b, "4bit", budget_gb) or "không vừa"))
        lines.append(f"| {model.name} | {model.kind} | {'?' if model.params_b is None else f'{model.params_b:g}'} | " + " | ".join(cell(values[key]) for key in ("bf16", "8bit", "4bit", "lora_bf16", "qlora_4bit")) + longest + " |")
    if any(model.params_b is None for model in models): lines += ["", "Dấu ? là model chưa khai báo params_b trong danh sách model."]
    lines += ["", f"Cách tính: trọng số = tỷ tham số × bit/tham số ÷ 8 (bf16 = 16, 8bit = 8,5, 4bit = 4,5 bit); train = trọng số × {TRAINING_FACTOR:g} + {CUDA_CONTEXT_GB:.0f} GB (batch 1, {REFERENCE_SEQ_LEN} token, bật gradient checkpointing); QLoRA cộng thêm {KBIT_OVERHEAD_GB:g} × √(tỷ tham số) GB cho embedding float32 và logits (đo trên Qwen3-4B; model 14B, 27B chưa đo). Độ dài khác {REFERENCE_SEQ_LEN} token: giả định {SEQ_SHARE:.0%} phần thêm khi train tăng theo số token (chưa đo). Suy luận cần thêm bộ nhớ cho KV cache, tăng theo độ dài ngữ cảnh."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m local_ai.models.vram", description="In bảng VRAM ước tính cho mọi model trong danh sách model (không tải model).")
    parser.add_argument("--models", default=str(ROOT / "configs" / "models" / "platform.json"), help="File danh sách model (mặc định configs/models/platform.json)")
    parser.add_argument("--seq-len", type=int, default=REFERENCE_SEQ_LEN, help=f"Độ dài tối đa mỗi dòng khi train, bằng max_length trong cấu hình train (mặc định {REFERENCE_SEQ_LEN} token)")
    parser.add_argument("--budget-gb", type=float, help="Thêm cột độ dài lớn nhất (bội số của 64 token) mà train QLoRA còn vừa số GB này, ví dụ 16")
    args = parser.parse_args(argv)
    if args.seq_len <= 0: parser.error(f"--seq-len phải lớn hơn 0, nhận được {args.seq_len}")
    if args.budget_gb is not None and args.budget_gb <= 0: parser.error(f"--budget-gb phải lớn hơn 0, nhận được {args.budget_gb:g}")
    print(format_table(load_model_configs(args.models), args.seq_len, args.budget_gb))


if __name__ == "__main__": main()
