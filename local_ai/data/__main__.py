from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from local_ai.data.core import build_dataset, deduplicate, load_records, statistics, validate_records, verify_math, verify_python, verify_tool_call, write_jsonl
from local_ai.data.secrets import scan_secrets


def _exit_after_stream() -> None:
    """Thư viện datasets (đã gặp với bản 5.0.1) còn luồng đọc ngầm khi dừng stream giữa chừng, làm Python crash lúc thoát
    ("Fatal Python error: PyGILState_Release"). Mọi file đã ghi xong nên thoát ngay, bỏ qua bước dọn dẹp của thư viện."""
    from local_ai.data.hub import real_stream_used
    if real_stream_used():
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m local_ai.data")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in (("inspect", "In 3 bản ghi đầu của file dữ liệu"), ("validate", "Kiểm tra schema, đếm bản ghi hợp lệ và bị loại"), ("deduplicate", "Đếm bản ghi trùng nội dung"), ("stats", "Thống kê domain, nguồn, độ dài")):
        item = commands.add_parser(name, help=text); item.add_argument("path", help="File dữ liệu (.jsonl, .json, .csv hoặc .parquet)")
    build = commands.add_parser("build", help="Kiểm tra, loại trùng, chặn trùng với eval rồi xuất train.jsonl, sft.jsonl, manifest.json"); build.add_argument("sources", nargs="+"); build.add_argument("--output", required=True); build.add_argument("--version", required=True); build.add_argument("--config", required=True); build.add_argument("--eval-source", action="append", default=[])
    generate = commands.add_parser("generate", help="Kiểm chứng bản ghi do model thầy sinh ra (math, python, tool_call)"); generate.add_argument("teacher_output", help="Các bản ghi JSONL do TeacherModel đã sinh ra"); generate.add_argument("--output", required=True); generate.add_argument("--verifier", choices=("math", "python", "tool_call"), required=True)
    hf = commands.add_parser("hf-sft", help="Tải dataset từ Hugging Face, kiểm tra rồi xuất sft.jsonl")
    source = hf.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="File cấu hình một dataset (ví dụ configs/datasets/presets/code.json)")
    source.add_argument("--preset", action="append", help="Tên preset, có thể kèm tỉ lệ trộn: --preset code:0.5 --preset vietnamese:0.5 (lặp lại để trộn nhiều preset)")
    hf.add_argument("--dataset", help="Ghi đè hf_dataset.name (chỉ dùng với --config)"); hf.add_argument("--output", help="Thư mục đầu ra (mặc định: output_dir trong cấu hình, hoặc data/processed/hf_sft khi trộn preset)")
    hf.add_argument("--limit", type=int, help="Số dòng tối đa đọc từ mỗi dataset"); hf.add_argument("--total", type=int, help="Tổng số dòng đọc khi trộn preset, chia theo tỉ lệ")
    commands.add_parser("list-presets", help="Liệt kê các preset dataset trong configs/datasets/presets/")
    commands.add_parser("secret-scan", help="Quét repo tìm khóa/mật khẩu bị lộ")
    args = parser.parse_args()
    if args.command == "list-presets":
        from local_ai.data.hub import list_presets
        for item in list_presets():
            print(f"{item['name']}: {item['dataset']}@{item['revision'][:8]} | domain {item['domain']} | ngôn ngữ {item['language']} | tối đa {item['limit']} dòng | giấy phép {item['license']} ({item['license_status']})\n    {item['description']}")
        return
    if args.command == "hf-sft" and args.preset:
        from local_ai.data.hub import DEFAULT_SFT_DIR, parse_mix, prepare_hf_mix
        print(json.dumps(prepare_hf_mix(parse_mix(args.preset), args.output or DEFAULT_SFT_DIR, total=args.total, limit=args.limit), indent=2, ensure_ascii=False)); _exit_after_stream(); return
    if args.command == "hf-sft":
        from local_ai.data.hub import prepare_hf_sft
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        if args.dataset: config["hf_dataset"]["name"] = args.dataset
        if args.limit: config["hf_dataset"]["limit"] = args.limit
        print(json.dumps(prepare_hf_sft(config, args.output), indent=2)); _exit_after_stream(); return
    if args.command == "secret-scan":
        findings = scan_secrets(Path(".")); print(json.dumps({"findings": findings}, indent=2));
        if findings: raise SystemExit("Phát hiện có thể lộ thông tin bí mật. Hãy đổi các khóa bị lộ và xóa chúng khỏi repo trước khi tiếp tục.")
        return
    if args.command == "generate":
        verifier = {"math": verify_math, "python": verify_python, "tool_call": verify_tool_call}[args.verifier]
        accepted, rejected = [], []
        for record in load_records(args.teacher_output):
            verification = verifier(record); record["verification"] = verification; record["validation_status"] = "valid" if verification["passed"] else "rejected"; (accepted if verification["passed"] else rejected).append(record)
        write_jsonl(args.output, accepted); write_jsonl(Path(args.output).with_name("rejected.jsonl"), rejected); print(json.dumps({"accepted": len(accepted), "rejected": len(rejected)})); return
    records = load_records(args.path) if args.command != "build" else None
    if args.command == "inspect": print(json.dumps(records[:3], indent=2)); return
    if args.command == "stats": print(json.dumps(statistics(records), indent=2)); return
    if args.command == "validate":
        valid, rejected = validate_records(records); print(json.dumps({"valid": len(valid), "rejected": len(rejected)}, indent=2)); return
    if args.command == "deduplicate":
        unique, duplicates = deduplicate(records); print(json.dumps({"unique": len(unique), "duplicates": len(duplicates)}, indent=2)); return
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    print(json.dumps(build_dataset(args.sources, args.output, args.version, config, args.eval_source), indent=2))

if __name__ == "__main__": main()
