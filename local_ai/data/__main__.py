from __future__ import annotations

import argparse
import json
from pathlib import Path

from local_ai.data.core import build_dataset, deduplicate, load_records, statistics, validate_records, verify_math, verify_python, verify_tool_call, write_jsonl
from local_ai.data.corpus import Registry, Source, Tokenizer, build_one, check_source_domains, domain_mixture, domain_targets, scan_secrets


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m local_ai.data")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "validate", "deduplicate", "stats"):
        item = commands.add_parser(name); item.add_argument("path")
    build = commands.add_parser("build"); build.add_argument("sources", nargs="+"); build.add_argument("--output", required=True); build.add_argument("--version", required=True); build.add_argument("--config", required=True); build.add_argument("--eval-source", action="append", default=[])
    generate = commands.add_parser("generate"); generate.add_argument("teacher_output", help="Các bản ghi JSONL do TeacherModel đã sinh ra"); generate.add_argument("--output", required=True); generate.add_argument("--verifier", choices=("math", "python", "tool_call"), required=True)
    hf = commands.add_parser("hf-sft", help="Tải dataset từ Hugging Face, kiểm tra rồi xuất sft.jsonl"); hf.add_argument("--config", required=True); hf.add_argument("--dataset", help="Ghi đè hf_dataset.name"); hf.add_argument("--output"); hf.add_argument("--limit", type=int)
    for name in ("sources", "mixture", "acquire", "build-corpus", "resume", "progress", "tokenizer-info", "secret-scan"):
        item = commands.add_parser(name); item.add_argument("--config", required=name not in {"secret-scan"}); item.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "hf-sft":
        from local_ai.data.hub import prepare_hf_sft
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        if args.dataset: config["hf_dataset"]["name"] = args.dataset
        if args.limit: config["hf_dataset"]["limit"] = args.limit
        print(json.dumps(prepare_hf_sft(config, args.output), indent=2)); return
    if args.command in {"sources", "mixture", "acquire", "build-corpus", "resume", "progress", "tokenizer-info"}:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        sources = [Source.from_dict(item) for item in config.get("sources", [])]
        if args.command == "mixture": print(json.dumps({"target_tokens": config["target_tokens"], "domain_mixture": domain_mixture(config), "domain_targets": domain_targets(config)}, indent=2)); return
        check_source_domains(sources, config)
        if args.command == "sources": print(json.dumps([source.__dict__ for source in sources], indent=2)); return
        if args.command == "tokenizer-info": print(json.dumps(Tokenizer(config["tokenizer"]).info(), indent=2)); return
        registry = Registry(Path(config["storage_root"]))
        if args.command == "progress": print(json.dumps(registry.progress(config["target_tokens"], domain_mixture(config)), indent=2)); return
        if args.command == "resume":
            incomplete = registry.db.execute("SELECT shard_id, status FROM shards WHERE status != 'COMPLETE'").fetchall()
            results = [build_one(source, config, args.dry_run) for source in sources] if incomplete else []
            print(json.dumps({"incomplete_shards": incomplete, "resumed": results}, indent=2)); return
        selected = sources[:1] if args.command == "acquire" else sources
        results = [build_one(source, config, args.dry_run) for source in selected]
        print(json.dumps({"results": results, "progress": registry.progress(config["target_tokens"], domain_mixture(config))}, indent=2)); return
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
