"""Lệnh dữ liệu: `python -m local_ai.data <lệnh>`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from local_ai.data.core import (
    build_dataset,
    deduplicate,
    load_records,
    statistics,
    validate_records,
    verify_math,
    verify_python,
    verify_tool_call,
    write_jsonl,
)
from local_ai.data.secrets import scan_secrets


def print_json(value: object) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m local_ai.data")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in (
        ("inspect", "Xem 3 bản ghi đầu"),
        ("validate", "Đếm bản ghi hợp lệ/bị loại"),
        ("deduplicate", "Đếm bản ghi trùng"),
        ("stats", "Thống kê dữ liệu"),
    ):
        commands.add_parser(name, help=text).add_argument("path")

    build = commands.add_parser("build", help="Kiểm tra, loại trùng, chặn rò rỉ eval rồi xuất sft.jsonl")
    build.add_argument("sources", nargs="+")
    build.add_argument("--output", required=True)
    build.add_argument("--version", required=True)
    build.add_argument("--config", required=True)
    build.add_argument("--eval-source", action="append", default=[])

    generate = commands.add_parser("generate", help="Lọc dữ liệu tổng hợp bằng bộ kiểm chứng")
    generate.add_argument("teacher_output", help="Các bản ghi JSONL do TeacherModel đã sinh ra")
    generate.add_argument("--output", required=True)
    generate.add_argument("--verifier", choices=("math", "python", "tool_call"), required=True)

    hf = commands.add_parser("hf-sft", help="Tải dataset từ Hugging Face, kiểm tra rồi xuất sft.jsonl")
    hf.add_argument("--config", required=True)
    hf.add_argument("--dataset", help="Ghi đè hf_dataset.name")
    hf.add_argument("--output")
    hf.add_argument("--limit", type=int)

    commands.add_parser("secret-scan", help="Quét repo tìm khóa/mật khẩu bị lộ")
    args = parser.parse_args()

    if args.command == "hf-sft":
        from local_ai.data.hub import prepare_hf_sft

        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        if args.dataset:
            config["hf_dataset"]["name"] = args.dataset
        if args.limit:
            config["hf_dataset"]["limit"] = args.limit
        print_json(prepare_hf_sft(config, args.output))
    elif args.command == "secret-scan":
        findings = scan_secrets(Path("."))
        print_json({"findings": findings})
        if findings:
            raise SystemExit(
                "Phát hiện có thể lộ thông tin bí mật. Hãy đổi các khóa bị lộ và xóa chúng khỏi repo trước khi tiếp tục."
            )
    elif args.command == "generate":
        verifier = {"math": verify_math, "python": verify_python, "tool_call": verify_tool_call}[args.verifier]
        accepted, rejected = [], []
        for record in load_records(args.teacher_output):
            record["verification"] = verifier(record)
            passed = record["verification"]["passed"]
            record["validation_status"] = "valid" if passed else "rejected"
            (accepted if passed else rejected).append(record)
        write_jsonl(args.output, accepted)
        write_jsonl(Path(args.output).with_name("rejected.jsonl"), rejected)
        print_json({"accepted": len(accepted), "rejected": len(rejected)})
    elif args.command == "build":
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        print_json(build_dataset(args.sources, args.output, args.version, config, args.eval_source))
    else:
        records = load_records(args.path)
        if args.command == "inspect":
            print_json(records[:3])
        elif args.command == "stats":
            print_json(statistics(records))
        elif args.command == "validate":
            valid, rejected = validate_records(records)
            print_json({"valid": len(valid), "rejected": len(rejected)})
        else:
            unique, duplicates = deduplicate(records)
            print_json({"unique": len(unique), "duplicates": len(duplicates)})


if __name__ == "__main__":
    main()
