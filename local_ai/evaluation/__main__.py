"""Lệnh chạy eval: `python -m local_ai.evaluation --model <tên>` (hoặc `--scripted` để chạy thử bằng đáp án mẫu)."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from datetime import datetime
from pathlib import Path

from local_ai.config.settings import find_model_config
from local_ai.evaluation.suite import DEFAULT_CASES, check_train_overlap, load_cases, run_eval, write_reports
from local_ai.models.router import ScriptedModelAdapter, create_adapter

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m local_ai.evaluation", description="Chạy bộ eval cho một model, ghi báo cáo JSON và Markdown vào .runs/.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--model", help="Tên model trong danh sách model (ví dụ smoke, ollama)")
    target.add_argument("--scripted", action="store_true", help="Chạy thử bằng model giả trả lời đúng đáp án mẫu của từng câu (không cần model thật)")
    parser.add_argument("--models", default=str(ROOT / "configs" / "models" / "platform.json"), help="File danh sách model")
    parser.add_argument("--cases", default=str(DEFAULT_CASES), help="File câu hỏi eval (JSONL)")
    parser.add_argument("--output", help="Thư mục ghi báo cáo (mặc định .runs/eval/<model>-<thời điểm>)")
    parser.add_argument("--train-data", action="append", default=[], help="File dữ liệu train (train.jsonl hoặc sft.jsonl) cần kiểm tra trùng với eval; lặp lại được")
    args = parser.parse_args(argv)

    cases = load_cases(args.cases)
    missing_files = [path for path in args.train_data if not Path(path).is_file()]
    if missing_files:
        print(f"Không tìm thấy file dữ liệu train: {', '.join(missing_files)}", file=sys.stderr)
        return 2
    overlap = check_train_overlap(cases, args.train_data)
    if overlap:
        print(f"Không chạy eval: dữ liệu train trùng với câu eval: {', '.join(f'{identifier} (trùng {reason})' for identifier, reason in overlap)}", file=sys.stderr)
        return 2
    if args.scripted:
        model = ScriptedModelAdapter("scripted-reference", [case.reference or "" for case in cases])
    else:
        config = find_model_config(args.models, args.model)
        missing = [name for name in ("torch", "transformers") if config.backend == "transformers" and importlib.util.find_spec(name) is None]
        if missing:
            print(f"Bỏ qua: thiếu thư viện {', '.join(missing)} để chạy model '{config.name}' (status skipped).", file=sys.stderr)
            return 0
        model = create_adapter(config)
    report = run_eval(model, cases)
    output = Path(args.output or ROOT / ".runs" / "eval" / f"{model.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    json_path, markdown_path = write_reports(report, output)
    if report["status"] == "completed": print(f"{report['model']}: {report['passed']}/{report['cases']} câu đạt ({report['accuracy']:.0%}).")
    else: print(report["error"], file=sys.stderr)
    print(f"Báo cáo: {json_path} và {markdown_path}")
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
