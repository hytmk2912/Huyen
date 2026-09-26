"""Ước tính thời gian train và chấm eval trên GPU của Colab (mặc định T4), chỉ từ cấu hình và file dữ liệu: không tải model.

Đây là ước lượng thô. Thông lượng train đã sửa theo 2 lần đo thật trên T4 (`smoke` và `light`, 25/9/2026); tốc độ chấm
chưa đo được (xem `python -m local_ai.training.calibrate`). Cách tính:
- số bước = ceil(số dòng ÷ (per_device_batch_size × gradient_accumulation_steps)) × epochs, hoặc max_steps nếu lớn hơn 0;
- số token mỗi dòng ≈ số ký tự ÷ CHARS_PER_TOKEN + TEMPLATE_TOKENS × số lượt hội thoại, không quá max_length.
  Hai hằng số này khớp với tokenizer Qwen3 trên 2000 dòng thật của 3 preset (đo ngày 25/9/2026: trung bình 491 token/dòng
  khi cắt ở 1024 token, 552 khi cắt ở 2048);
- batch lớn hơn 1 thì mỗi dòng được đệm (padding) tới dòng dài nhất trong batch, nên tính theo dòng dài nhất
  (xếp ngẫu nhiên các dòng thành batch như lúc train);
- chưa có file dữ liệu thì dùng DEFAULT_ROWS dòng và số token mỗi dòng đo sẵn trong MEASURED_TOKENS_PER_ROW;
- phép tính khi train ≈ 2 × số tham số × số token × số lượt chạy qua model (3 khi bật gradient checkpointing: chiều xuôi,
  tính lại, chiều ngược; 2 khi không bật). Thời gian train = phép tính ÷ thông lượng hiệu dụng của GPU;
- thời gian chấm (trường hợp xấu nhất: câu nào cũng sinh đủ max_new_tokens) = số câu × max_new_tokens × thời gian sinh
  1 token, với thời gian sinh 1 token = token_overhead_s + (2 byte × số tham số) ÷ băng thông bộ nhớ hiệu dụng.
Chưa tính thời gian cài thư viện, tải model và tải dữ liệu (thường vài phút).
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_ai.evaluation.suite import load_cases
from local_ai.models.vram import training_gb
from local_ai.training.finetune import FinetuneConfig, effective_quantization, load_finetune_config, resolve_base_model, resume_target

CHARS_PER_TOKEN = 3.4
TEMPLATE_TOKENS = 5  # token của chat template cho mỗi lượt hội thoại (<|im_start|>, vai trò, xuống dòng...)
DEFAULT_ROWS = 2000  # số dòng notebook Colab lấy từ preset
# Số token mỗi dòng (kể cả phần đệm) đo trên 2000 dòng thật của 3 preset, theo (max_length, per_device_batch_size).
MEASURED_TOKENS_PER_ROW = {(1024, 1): 490, (1024, 4): 820, (2048, 1): 558, (2048, 4): 1062}
UNCUT_TOKENS_PER_ROW = 562  # trung bình khi không cắt, batch 1: dùng cho cặp chưa đo


@dataclass(frozen=True)
class GpuProfile:
    name: str
    memory_gb: float
    train_tflops: float  # thông lượng hiệu dụng khi train LoRA/QLoRA fp16
    memory_gbps: float  # băng thông bộ nhớ hiệu dụng khi sinh chữ
    token_overhead_s: float  # thời gian cố định cho mỗi token khi sinh bằng transformers (gọi kernel, Python)


# T4: đỉnh lý thuyết 65 TFLOPS fp16, băng thông 320 GB/s, khoảng 15 GB dùng được.
# - train_tflops: đo thật khi train QLoRA 4bit, fp16 ngày 25/9/2026: `smoke` 5,36 TFLOPS (125 bước trong 14,9 phút),
#   `light` 5,17 TFLOPS (95 bước trong 66,7 phút, khoảng 42 giây/bước). Lấy 5,2 để ước tính của `light` (lần train dài nhất
#   trên Colab) sát nhất; `smoke` lệch khoảng 3%. Số đo lưu ở tests/fixtures/measurements/that_*_t4_2026-09-25.json.
#   Trước đó giả định 6 (khoảng 10% đỉnh).
# - memory_gbps, token_overhead_s: giả định (sinh chữ lấy 60% băng thông), chưa đo được vì báo cáo chấm của 2 lần chạy trên
#   tính cả thời gian nạp model. Dù vậy, lần chấm `light` (câu nào cũng sinh gần đủ 512 token) mất 17,4 phút, khớp ước tính 17,2.
GPUS = {"T4": GpuProfile("T4", memory_gb=15.0, train_tflops=5.2, memory_gbps=190.0, token_overhead_s=0.025)}


def row_tokens(row: dict[str, Any], max_length: int) -> int:
    messages = row.get("messages") or []
    chars = sum(len(str(message.get("content") or "")) for message in messages)
    return min(max_length, math.ceil(chars / CHARS_PER_TOKEN) + TEMPLATE_TOKENS * len(messages))


def padded_average(counts: list[int], batch: int, seed: int) -> float:
    """Số token trung bình mỗi dòng khi xếp ngẫu nhiên thành batch và đệm tới dòng dài nhất của batch."""
    order = counts[:]; random.Random(seed).shuffle(order)
    return sum(max(order[start:start + batch]) * len(order[start:start + batch]) for start in range(0, len(order), batch)) / len(order)


def dataset_stats(config: FinetuneConfig) -> tuple[int, float] | None:
    """(số dòng, số token trung bình mỗi dòng kể cả phần đệm) của sft.jsonl; chưa có file thì trả None."""
    file = Path(config.dataset_path)
    if not file.is_file(): return None
    counts = [row_tokens(json.loads(line), config.max_length) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
    return (len(counts), padded_average(counts, config.per_device_batch_size, config.seed)) if counts else None


def training_steps(config: FinetuneConfig, rows: int) -> int:
    if config.max_steps > 0: return config.max_steps
    return math.ceil(rows / (config.per_device_batch_size * config.gradient_accumulation_steps)) * config.epochs


def seconds_per_generated_token(params_b: float, gpu: GpuProfile) -> float:
    return gpu.token_overhead_s + 2 * params_b / gpu.memory_gbps  # 2 byte mỗi tham số fp16; params_b tỷ ÷ GB/s


def local_trainer_state(config: FinetuneConfig) -> dict[str, Any] | None:
    """trainer_state.json của checkpoint mà lệnh train sẽ chạy tiếp (checkpoint-N hoặc last-checkpoint đã tải về)."""
    target = resume_target(config)
    state = Path(target) / "trainer_state.json" if target else None
    return json.loads(state.read_text(encoding="utf-8")) if state and state.is_file() else None


def estimate(config: FinetuneConfig, rows: int | None = None, max_new_tokens: int | None = None, eval_cases: int | None = None,
             trainer_state: dict[str, Any] | None = None, gpu: str = "T4") -> dict[str, Any]:
    """Ước tính số bước, số phút train (còn lại nếu đã có checkpoint), số phút chấm eval và VRAM so với GPU."""
    profile, model = GPUS[gpu], resolve_base_model(config)
    if model.params_b is None: raise ValueError(f"Model '{model.name}' chưa khai báo params_b trong danh sách model, không ước tính được")
    stats = dataset_stats(config)
    if stats and rows is None: rows, tokens_per_row, source = stats[0], stats[1], "sft.jsonl"
    else: rows, tokens_per_row, source = rows or DEFAULT_ROWS, MEASURED_TOKENS_PER_ROW.get((config.max_length, config.per_device_batch_size), min(config.max_length, UNCUT_TOKENS_PER_ROW)), "đo mẫu"
    steps = training_steps(config, rows)
    tokens_per_step = config.per_device_batch_size * config.gradient_accumulation_steps * tokens_per_row
    tokens = steps * tokens_per_step
    passes = 3 if config.gradient_checkpointing else 2
    train_minutes = 2 * model.params_b * 1e9 * tokens * passes / (profile.train_tflops * 1e12) / 60
    done = min(int((trainer_state or {}).get("global_step") or 0), steps)
    cases = eval_cases if eval_cases is not None else len(load_cases())
    new_tokens = max_new_tokens or model.max_new_tokens
    eval_minutes = cases * new_tokens * seconds_per_generated_token(model.params_b, profile) / 60
    quantization = effective_quantization(config, model)
    vram = training_gb(model.params_b, quantization or "bf16")
    return {"gpu": profile.name, "model": model.name, "source": model.source, "params_b": model.params_b, "rows": rows, "tokens_per_row": round(tokens_per_row, 1),
            "data": source, "max_length": config.max_length, "steps": steps, "tokens_per_step": round(tokens_per_step, 1), "done_steps": done, "train_minutes": round(train_minutes, 1),
            "remaining_train_minutes": round(train_minutes * (steps - done) / steps, 1), "eval_cases": cases, "max_new_tokens": new_tokens,
            "eval_minutes": round(eval_minutes, 1), "vram_gb": round(vram, 1), "gpu_memory_gb": profile.memory_gb,
            "fits": vram <= profile.memory_gb and config.max_length <= 2048}  # ước tính VRAM của vram.py giả định khoảng 2048 token


def _number(value: float) -> str:
    return f"{value:g}".replace(".", ",")


def format_estimate(plan: dict[str, Any], resumed_from: str | None = None) -> str:
    lines = [f"Ước tính cho model {plan['model']} ({plan['source']}, {_number(plan['params_b'])} tỷ tham số) trên GPU {plan['gpu']}. Đây là ước lượng thô; thông lượng train đo từ lần chạy thật trên T4 (smoke và light), VRAM và tốc độ chấm chưa sửa theo số đo.",
             f"- Dữ liệu: {plan['rows']} dòng, trung bình khoảng {round(plan['tokens_per_row'])} token/dòng kể cả phần đệm (cắt ở {plan['max_length']} token; {'đếm từ sft.jsonl' if plan['data'] == 'sft.jsonl' else 'chưa có sft.jsonl nên dùng số đo mẫu'}).",
             f"- VRAM khi train: khoảng {_number(plan['vram_gb'])} GB / {_number(plan['gpu_memory_gb'])} GB của {plan['gpu']}." + ("" if plan["fits"] else " CẢNH BÁO: có thể thiếu bộ nhớ, hãy giảm max_length hoặc chọn model nhỏ hơn."),
             f"- Train: {plan['steps']} bước, khoảng {round(plan['train_minutes'])} phút."]
    if plan["done_steps"]:
        lines.append(f"- Đã train {plan['done_steps']}/{plan['steps']} bước (checkpoint {resumed_from or 'đã lưu'}): lệnh train sẽ tự chạy tiếp, còn khoảng {round(plan['remaining_train_minutes'])} phút.")
    lines.append(f"- Chấm eval: tối đa khoảng {round(plan['eval_minutes'])} phút mỗi lần ({plan['eval_cases']} câu × {plan['max_new_tokens']} token); notebook chấm 2 lần (trước và sau khi train).")
    total = plan["remaining_train_minutes"] + 2 * plan["eval_minutes"]
    lines.append(f"- Tổng còn lại, chưa tính cài thư viện và tải model (thường thêm 5–10 phút): khoảng {round(total)} phút.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m local_ai.training.estimate", description="Ước tính thời gian train và chấm eval trên GPU của Colab (không tải model; ước lượng thô).")
    parser.add_argument("--config", required=True, help="File cấu hình huấn luyện (JSON)")
    parser.add_argument("--rows", type=int, help="Số dòng dữ liệu dùng để ước tính (mặc định: đếm trong sft.jsonl, chưa có thì 2000)")
    parser.add_argument("--max-new-tokens", type=int, help="Số token tối đa mỗi câu khi chấm eval (mặc định theo danh sách model)")
    parser.add_argument("--hub-model-id", help="Repo Hugging Face chứa checkpoint (tên-người-dùng/tên-repo) để xem đã train tới bước nào")
    parser.add_argument("--gpu", choices=sorted(GPUS), default="T4", help="Loại GPU (mặc định T4 của Colab miễn phí)")
    parser.add_argument("--dry-run", action="store_true", help="Không gọi Hugging Face, chỉ ước tính từ cấu hình và file trên máy")
    args = parser.parse_args(argv)
    config = load_finetune_config(args.config)
    state, resumed_from = local_trainer_state(config), "trên máy"
    if state is None and args.hub_model_id and not args.dry_run:
        from local_ai.training.hub import remote_trainer_state
        try:
            state, resumed_from = remote_trainer_state(args.hub_model_id), "trên Hugging Face"
        except Exception as error:  # chỉ để xem tiến độ: lỗi mạng hay quyền không được chặn cả notebook
            print(f"Không đọc được tiến độ trên Hugging Face ({type(error).__name__}); vẫn ước tính như train từ đầu.")
    print(format_estimate(estimate(config, rows=args.rows, max_new_tokens=args.max_new_tokens, trainer_state=state, gpu=args.gpu), resumed_from))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
