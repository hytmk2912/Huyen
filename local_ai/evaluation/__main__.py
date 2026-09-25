"""Lệnh chạy eval: `python -m local_ai.evaluation --model <tên>` (hoặc `--scripted` để chạy thử bằng đáp án mẫu)."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

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
    parser.add_argument("--max-new-tokens", type=int, help="Ghi đè số token tối đa model được sinh cho mỗi câu (nhỏ hơn thì chấm nhanh hơn)")
    parser.add_argument("--hub-repo", help="Repo Hugging Face riêng tư lưu báo cáo (tên-người-dùng/tên-repo): chấm xong thì đẩy lên; chạy lại mà repo đã có báo cáo cùng cài đặt thì dùng lại, không chấm lại")
    parser.add_argument("--hub-path", default="eval", help="Thư mục trong repo chứa report.json (mặc định eval)")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ kiểm tra tham số và in kế hoạch, không nạp model")
    args = parser.parse_args(argv)
    if args.hub_repo:
        from local_ai.training.hub import check_repo_id
        check_repo_id(args.hub_repo)

    cases = load_cases(args.cases)
    if args.dry_run:
        plan = {"status": "dry-run", "cases": len(cases), "output": args.output or ".runs/eval/<model>-<thời điểm>", "train_data": {path: Path(path).is_file() for path in args.train_data}}
        if not args.scripted:
            config = find_model_config(args.models, args.model)
            plan["model"] = {"name": config.name, "backend": config.backend, "source": config.source, "adapter_path": config.adapter_path, "max_new_tokens": args.max_new_tokens or config.max_new_tokens}
        if args.hub_repo: plan["hub"] = {"repo": args.hub_repo, "path": f"{args.hub_path}/report.json"}
        print(json.dumps(plan, ensure_ascii=False, indent=2)); return 0
    missing_files = [path for path in args.train_data if not Path(path).is_file()]
    if missing_files:
        print(f"Không tìm thấy file dữ liệu train: {', '.join(missing_files)}", file=sys.stderr)
        return 2
    overlap = check_train_overlap(cases, args.train_data)
    if overlap:
        print(f"Không chạy eval: dữ liệu train trùng với câu eval: {', '.join(f'{identifier} (trùng {reason})' for identifier, reason in overlap)}", file=sys.stderr)
        return 2
    config = None if args.scripted else find_model_config(args.models, args.model)
    if config is not None and args.max_new_tokens: config = replace(config, max_new_tokens=args.max_new_tokens)
    name = "scripted-reference" if config is None else config.name
    output = Path(args.output or ROOT / ".runs" / "eval" / f"{name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    settings = eval_settings(name, None if config is None else config.max_new_tokens, args.cases)
    if args.hub_repo and reuse_hub_report(args.hub_repo, args.hub_path, settings, output): return 0
    if config is None:
        model = ScriptedModelAdapter("scripted-reference", [case.reference or "" for case in cases])
    else:
        missing = [library for library in ("torch", "transformers") if config.backend == "transformers" and importlib.util.find_spec(library) is None]
        if missing:
            print(f"Bỏ qua: thiếu thư viện {', '.join(missing)} để chạy model '{config.name}' (status skipped).", file=sys.stderr)
            return 0
        model = create_adapter(config)
    report = {**run_eval(model, cases), "settings": settings}
    json_path, markdown_path = write_reports(report, output)
    if report["status"] == "completed": print(f"{report['model']}: {report['passed']}/{report['cases']} câu đạt ({report['accuracy']:.0%}).")
    else: print(report["error"], file=sys.stderr)
    print(f"Báo cáo: {json_path} và {markdown_path}")
    if args.hub_repo and report["status"] == "completed": push_hub_report(args.hub_repo, args.hub_path, json_path, markdown_path)
    return 0 if report["status"] == "completed" else 1


REPORT_KEYS = {"model", "generated_at", "cases", "passed", "accuracy", "groups", "languages", "failures"}  # đủ để ghi report.md


def eval_settings(model: str, max_new_tokens: int | None, cases_path: str | Path) -> dict[str, Any]:
    """Những gì quyết định kết quả chấm: model, số token tối đa mỗi câu, nội dung bộ câu hỏi (mã băm)."""
    return {"model": model, "max_new_tokens": max_new_tokens, "cases_sha256": hashlib.sha256(Path(cases_path).read_bytes()).hexdigest()}


def reuse_hub_report(repo: str, hub_path: str, settings: dict[str, Any], output: Path) -> bool:
    """Hub đã có báo cáo chấm xong với đúng cài đặt thì ghi nó vào `output` và trả True (không phải chấm lại, ví dụ sau khi Colab ngắt)."""
    from local_ai.training.hub import download_file
    try:
        path = download_file(repo, f"{hub_path}/report.json")
    except Exception as error:  # mất mạng, sai quyền...: chấm lại cho chắc, không dừng notebook
        print(f"Không đọc được báo cáo trên Hugging Face ({type(error).__name__}); sẽ chấm lại.", file=sys.stderr); return False
    cached = json.loads(path.read_text(encoding="utf-8")) if path else None
    if not cached: return False
    if cached.get("status") != "completed" or cached.get("settings") != settings:
        print("Báo cáo trên Hugging Face dùng cài đặt khác (model, max_new_tokens hoặc bộ câu hỏi); sẽ chấm lại."); return False
    if not REPORT_KEYS <= set(cached):
        print("Báo cáo trên Hugging Face thiếu thông tin; sẽ chấm lại."); return False
    json_path, _ = write_reports(cached, output)
    print(f"Dùng lại báo cáo đã chấm trên Hugging Face ({repo}/{hub_path}): {cached['passed']}/{cached['cases']} câu đạt. Bỏ qua bước chấm. Báo cáo: {json_path}")
    return True


def push_hub_report(repo: str, hub_path: str, *paths: Path) -> None:
    from local_ai.training.hub import upload_file
    try:
        for path in paths: upload_file(repo, path, f"{hub_path}/{path.name}")
        print(f"Đã lưu báo cáo lên repo riêng tư {repo} ({hub_path}/): chạy lại notebook sẽ không phải chấm lại.")
    except Exception as error:  # không đẩy được thì lần sau chấm lại, không làm hỏng bước chấm vừa xong
        print(f"Không đẩy được báo cáo lên Hugging Face ({type(error).__name__}: {error}); lần chạy lại sẽ phải chấm lại.", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
