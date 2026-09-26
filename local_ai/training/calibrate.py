"""So số đo thật của một lần chạy notebook (train, chấm eval, agent) với ước tính, in bảng và hằng số đề xuất (mốc M15).

Số đo lấy từ:
- `measurements.json` do lệnh train ghi trong output_dir: GPU, VRAM đỉnh, thời gian train, bước bắt đầu và kết thúc;
- `report.json` của lệnh eval: thời gian chấm (`duration_s`) và tổng độ dài câu trả lời (`output_chars`);
- `report.json` của lệnh chạy nhiệm vụ agent (tùy chọn): tỉ lệ thành công và thời gian.

Hằng số đề xuất được tính ngược từ số đo, cùng công thức với `estimate.py` và `vram.py`:
- `train_tflops` = 2 × số tham số × số token đã train × số lượt chạy qua model ÷ số giây train;
- model nén (QLoRA): `KBIT_OVERHEAD_GB` = (VRAM đỉnh do torch đếm − GB trọng số × TRAINING_FACTOR) ÷ √(tỷ tham số);
  model không nén: `TRAINING_FACTOR` = VRAM đỉnh do torch đếm ÷ GB trọng số. Torch không đếm phần CUDA context, nên không
  trừ CUDA_CONTEXT_GB (ước tính vì vậy hơi cao hơn số đo, an toàn khi chọn GPU);
- `token_overhead_s` = số giây mỗi token khi chấm − thời gian đọc trọng số cho mỗi token. Báo cáo chấm cũ chưa có `load_s`
  (như lần chạy `smoke` ngày 25/9) tính cả thời gian tải và nạp model vào `duration_s`, nên số này bị đánh dấu là không dùng được.

Lệnh chỉ đề xuất, không tự sửa code: người sửa hằng số đọc bảng này rồi quyết định.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from local_ai.models.vram import KBIT_OVERHEAD_GB, TRAINING_FACTOR, weights_gb
from local_ai.training.estimate import CHARS_PER_TOKEN, GPUS, estimate, seconds_per_generated_token
from local_ai.training.finetune import MEASUREMENTS, FinetuneConfig, effective_quantization, load_finetune_config, resolve_base_model


MIN_WEIGHTS_GB = 1.0  # trọng số nhỏ hơn thì hằng số VRAM đo được không đại diện cho model lớn


def gpu_profile(name: str | None) -> str | None:
    """Tên hồ sơ trong GPUS ứng với tên GPU đo được (ví dụ "Tesla T4" → "T4"); GPU chưa có hồ sơ thì trả None."""
    return next((key for key in GPUS if name and key.casefold() in name.casefold()), None)


def read_json(path: str | Path | None) -> dict[str, Any] | None:
    return json.loads(Path(path).read_text(encoding="utf-8")) if path and Path(path).is_file() else None


def compare(config: FinetuneConfig, measurements: dict[str, Any], eval_reports: dict[str, dict[str, Any]] | None = None,
            agent_report: dict[str, Any] | None = None, max_new_tokens: int | None = None) -> dict[str, Any]:
    """Đặt số đo thật cạnh ước tính; trả về các mục so sánh và hằng số đề xuất (chỉ khi GPU có hồ sơ trong GPUS)."""
    model = resolve_base_model(config)
    profile = gpu_profile(measurements.get("gpu"))
    gpu = GPUS[profile or "T4"]
    plan = estimate(config, max_new_tokens=max_new_tokens, eval_cases=None, gpu=profile or "T4")
    proposals: dict[str, dict[str, Any]] = {}

    train = None
    steps, seconds = (measurements.get("end_step") or 0) - (measurements.get("start_step") or 0), measurements.get("train_seconds")
    if steps > 0 and seconds:
        passes = 3 if config.gradient_checkpointing else 2
        tflops = 2 * model.params_b * 1e9 * plan["tokens_per_step"] * steps * passes / seconds / 1e12
        train = {"steps": steps, "estimated_minutes": round(plan["train_minutes"] * steps / plan["steps"], 1), "measured_minutes": round(seconds / 60, 1), "effective_tflops": round(tflops, 3)}
        if profile: proposals["train_tflops"] = {"current": gpu.train_tflops, "measured": round(tflops, 2)}

    measured_vram = measurements.get("peak_reserved_gb") or measurements.get("peak_vram_gb")
    precision = effective_quantization(config, model) or "bf16"
    weights = weights_gb(model.params_b, precision)
    vram = {"estimated_gb": plan["vram_gb"], "measured_gb": measured_vram, "implied_training_factor": round(measured_vram / weights, 2) if measured_vram else None}
    if profile and measured_vram:
        # Model nhỏ: bộ nhớ cho logits/activation (tỉ lệ với batch × max_length × số từ vựng) lớn hơn trọng số nhiều lần,
        # nên hằng số VRAM tính từ model nhỏ không dùng được cho model lớn.
        note = f"đo với max_length {config.max_length}, batch {config.per_device_batch_size}" + ("" if weights >= MIN_WEIGHTS_GB else f"; trọng số chỉ {weights:.2f} GB nên KHÔNG dùng số này để sửa hệ số, hãy dùng số đo của model lớn hơn (ví dụ light)")
        if precision in ("4bit", "8bit"):
            overhead = (measured_vram - weights * TRAINING_FACTOR) / math.sqrt(model.params_b)
            proposals["KBIT_OVERHEAD_GB"] = {"current": KBIT_OVERHEAD_GB, "measured": round(overhead, 2), "note": note, "reliable": weights >= MIN_WEIGHTS_GB}
        else:
            proposals["TRAINING_FACTOR"] = {"current": TRAINING_FACTOR, "measured": vram["implied_training_factor"], "note": note, "reliable": weights >= MIN_WEIGHTS_GB}

    evaluations, overheads, clean = {}, [], True
    for label, report in (eval_reports or {}).items():
        if not report or report.get("status") != "completed" or not report.get("duration_s") or not report.get("output_chars"): continue
        per_token = report["duration_s"] / (report["output_chars"] / CHARS_PER_TOKEN)
        evaluations[label] = {"model": report["model"], "passed": report["passed"], "cases": report["cases"], "measured_minutes": round(report["duration_s"] / 60, 1),
                              "estimated_max_minutes": plan["eval_minutes"], "seconds_per_token": round(per_token, 4), "estimated_seconds_per_token": round(seconds_per_generated_token(model.params_b, gpu), 4)}
        overheads.append(per_token - 2 * model.params_b / gpu.memory_gbps); clean = clean and "load_s" in report
    if profile and overheads:
        proposals["token_overhead_s"] = {"current": gpu.token_overhead_s, "measured": round(max(0.0, sum(overheads) / len(overheads)), 4), "reliable": clean}
        if not clean: proposals["token_overhead_s"]["note"] = "báo cáo chấm cũ tính cả thời gian tải và nạp model, nên KHÔNG dùng số này để sửa hằng số"

    agent = None
    if agent_report and agent_report.get("status") == "completed":
        agent = {"passed": agent_report["passed"], "total": agent_report["total"], "success_rate": agent_report["success_rate"], "measured_minutes": round((agent_report.get("duration_s") or 0) / 60, 1)}
    return {"model": model.name, "gpu": measurements.get("gpu"), "profile": profile, "train": train, "vram": vram, "eval": evaluations, "agent": agent, "proposals": proposals, "measurements": measurements}


def _number(value: Any, digits: int = 1) -> str:
    return "—" if value is None else f"{value:.{digits}f}".replace(".", ",")


def format_comparison(result: dict[str, Any]) -> str:
    lines = [f"Số đo thật của model {result['model']} trên {result['gpu']}, so với ước tính của estimate.py và vram.py:", "", "| Mục | Ước tính | Đo thật |", "| --- | ---: | ---: |"]
    if result["train"]: lines.append(f"| Train {result['train']['steps']} bước (phút) | {_number(result['train']['estimated_minutes'])} | {_number(result['train']['measured_minutes'])} |")
    lines.append(f"| VRAM khi train (GB) | {_number(result['vram']['estimated_gb'])} | {_number(result['vram']['measured_gb'])} |")
    names = {"before": "Chấm trước khi train", "after": "Chấm sau khi train"}
    for label, item in result["eval"].items():
        name = names.get(label, label)
        lines.append(f"| {name}: thời gian (phút) | tối đa {_number(item['estimated_max_minutes'])} | {_number(item['measured_minutes'])} |")
        lines.append(f"| {name}: giây mỗi token | {_number(item['estimated_seconds_per_token'], 3)} | {_number(item['seconds_per_token'], 3)} |")
    if result["eval"]: lines.append("| Điểm eval | | " + " → ".join(f"{item['passed']}/{item['cases']}" for item in result["eval"].values()) + " |")
    if result["agent"]: lines.append(f"| Agent: nhiệm vụ đạt | | {result['agent']['passed']}/{result['agent']['total']} ({_number(result['agent']['measured_minutes'])} phút) |")
    lines.append("")
    if not result["profile"]:
        lines.append(f"GPU {result['gpu']} chưa có hồ sơ ước tính (hiện chỉ có {', '.join(GPUS)}), nên chỉ ghi số đo, không đề xuất hằng số.")
    elif result["proposals"]:
        lines.append("Hằng số đề xuất (chưa sửa code; chụp màn hình bảng này gửi lại để sửa ước tính):")
        where = {"train_tflops": f'estimate.py GPUS["{result["profile"]}"].train_tflops', "TRAINING_FACTOR": "vram.py TRAINING_FACTOR", "KBIT_OVERHEAD_GB": "vram.py KBIT_OVERHEAD_GB", "token_overhead_s": f'estimate.py GPUS["{result["profile"]}"].token_overhead_s'}
        lines += [f"- {where[key]}: {item['current']:g} → {item['measured']:g}" + (f" ({item['note']})" if item.get("note") else "") for key, item in result["proposals"].items()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m local_ai.training.calibrate", description="So số đo thật (thời gian, VRAM, tốc độ chấm) với ước tính, in bảng và hằng số đề xuất.")
    parser.add_argument("--config", required=True, help="File cấu hình huấn luyện (JSON) đã dùng khi train")
    parser.add_argument("--measurements", help=f"File số đo do lệnh train ghi (mặc định: <output_dir>/{MEASUREMENTS})")
    parser.add_argument("--eval-before", help="report.json của lần chấm trước khi train")
    parser.add_argument("--eval-after", help="report.json của lần chấm sau khi train")
    parser.add_argument("--agent-report", help="report.json của lệnh chạy nhiệm vụ agent")
    parser.add_argument("--max-new-tokens", type=int, help="Số token tối đa mỗi câu đã dùng khi chấm (để ước tính cho khớp)")
    parser.add_argument("--output", help="Nơi lưu số đo và so sánh (mặc định .runs/so_do/<model>.json)")
    parser.add_argument("--push-to-hub", action="store_true", help="Đẩy file số đo lên repo Hugging Face riêng tư (token lấy từ biến môi trường HF_TOKEN)")
    parser.add_argument("--hub-model-id", help="Repo nhận file số đo, dạng tên-người-dùng/tên-repo")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ kiểm tra tham số và các file số đo có hay chưa, không đẩy lên Hub")
    args = parser.parse_args(argv)
    config = load_finetune_config(args.config)
    if args.push_to_hub:
        from local_ai.training.hub import check_repo_id
        check_repo_id(args.hub_model_id)
    measurements_path = Path(args.measurements or Path(config.output_dir) / MEASUREMENTS)
    output = Path(args.output or Path(".runs") / "so_do" / f"{config.base_model}.json")
    files = {"measurements": measurements_path, "eval_before": args.eval_before, "eval_after": args.eval_after, "agent_report": args.agent_report}
    if args.dry_run:
        print(json.dumps({"status": "dry-run", "files": {name: {"path": str(path), "exists": Path(path).is_file()} for name, path in files.items() if path}, "output": str(output), "push_to_hub": args.hub_model_id if args.push_to_hub else None}, ensure_ascii=False, indent=2))
        return 0
    measurements = read_json(measurements_path)
    if measurements is None:
        print(f"Chưa có số đo: không thấy {measurements_path}; hãy chạy lệnh train trước", file=sys.stderr); return 2
    result = compare(config, measurements, {"before": read_json(args.eval_before), "after": read_json(args.eval_after)}, read_json(args.agent_report), args.max_new_tokens)
    print(format_comparison(result))
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nĐã lưu số đo vào {output}")
    if args.push_to_hub:
        from local_ai.training.hub import upload_file
        print("Đã đẩy số đo lên", upload_file(args.hub_model_id, output, f"so_do/{config.base_model}.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
