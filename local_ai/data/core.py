from __future__ import annotations

import ast
import csv
import hashlib
import json
import random
import unicodedata
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from local_ai.data.schema import DatasetExample, VALID_DOMAINS, VALID_STATUSES

REQUIRED = ("id", "domain", "task", "input", "expected_output", "source", "license", "dataset_version")
MAX_CHARS = 100_000


class TeacherModel(Protocol):
    def generate(self, prompt: str) -> str: ...
    def generate_batch(self, prompts: list[str]) -> list[str]: ...


def canonical_content(record: dict[str, Any]) -> str:
    fields = {key: record.get(key, "") for key in ("domain", "task", "input", "context", "expected_output", "reasoning", "trajectory", "messages")}
    return json.dumps(fields, sort_keys=True, separators=(",", ":"))


def content_hash(record: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_content(record).encode()).hexdigest()


def normalize_text(text: Any) -> str:
    """Chuẩn hóa để so sánh: Unicode NFC (chữ tiếng Việt dựng sẵn và tổ hợp coi như nhau), bỏ phân biệt hoa thường, gộp khoảng trắng."""
    return " ".join(unicodedata.normalize("NFC", str(text or "")).casefold().split())


def _questions(record: dict[str, Any]) -> set[str]:
    """Các câu hỏi của một bản ghi: trường input hoặc prompt, và mọi lượt user trong messages."""
    turns = [message.get("content") for message in record.get("messages") or [] if isinstance(message, dict) and message.get("role") == "user"]
    return {normalize_text(text).strip(" .?!") for text in (record.get("input"), record.get("prompt"), *turns)} - {""}


def find_eval_overlap(records: list[dict[str, Any]], eval_records: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Bản ghi train trùng với eval: cùng id, cùng nội dung (content_hash) hoặc cùng câu hỏi sau khi chuẩn hóa. Trả về (id, lý do)."""
    ids = {str(record["id"]) for record in eval_records if record.get("id")}
    hashes = {content_hash(record) for record in eval_records}
    questions = set().union(*(_questions(record) for record in eval_records)) if eval_records else set()
    overlap = []
    for record in records:
        reason = "id" if str(record.get("id")) in ids else "nội dung" if content_hash(record) in hashes else "câu hỏi" if _questions(record) & questions else None
        if reason: overlap.append((str(record.get("id")), reason))
    return overlap


def load_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else payload["examples"]
    if suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    if suffix in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError("Đọc file parquet cần thư viện tùy chọn pyarrow; hãy cài nó để nạp file parquet") from error
        return pq.read_table(path).to_pylist()
    raise ValueError(f"Không hỗ trợ định dạng dữ liệu: {path.suffix}")


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")


MESSAGE_ROLES = {"system", "user", "assistant"}


def valid_messages(messages: Any) -> bool:
    """Hội thoại hợp lệ: mỗi lượt có role (system/user/assistant) và content không rỗng, có ít nhất một lượt user, lượt cuối là assistant."""
    if not isinstance(messages, list) or not messages: return False
    if not all(isinstance(m, dict) and m.get("role") in MESSAGE_ROLES and isinstance(m.get("content"), str) and m["content"].strip() for m in messages): return False
    return any(m["role"] == "user" for m in messages) and messages[-1]["role"] == "assistant"


def sft_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    """Hội thoại cho sft.jsonl: giữ nguyên messages nếu có (kể cả system và mọi lượt); nếu không thì dựng từ
    input/expected_output, đưa context lên trước câu hỏi và reasoning vào khối <think>...</think> (định dạng của Qwen3)."""
    if record.get("messages"): return [{"role": m["role"], "content": m["content"]} for m in record["messages"]]
    question = f"{record['context']}\n\n{record['input']}" if record.get("context") else record["input"]
    answer = f"<think>\n{record['reasoning']}\n</think>\n\n{record['expected_output']}" if record.get("reasoning") else record["expected_output"]
    return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]


def validate_record(record: dict[str, Any], seen_ids: set[str] | None = None) -> list[str]:
    errors = [f"missing:{key}" for key in REQUIRED if key not in record or record[key] in (None, "")]
    if errors: return errors
    if not isinstance(record["source"], dict) or not record["source"].get("name"):
        errors.append("invalid:source")
    if not isinstance(record["license"], dict) or not record["license"].get("name"):
        errors.append("invalid:license")
    if record["domain"] not in VALID_DOMAINS: errors.append("invalid:domain")
    if record.get("validation_status", "pending") not in VALID_STATUSES: errors.append("invalid:validation_status")
    if not isinstance(record.get("difficulty", 1), int) or not 1 <= record.get("difficulty", 1) <= 5: errors.append("invalid:difficulty")
    if record.get("quality_score") is not None and not 0 <= record["quality_score"] <= 1: errors.append("invalid:quality_score")
    if record.get("messages") is not None and not valid_messages(record["messages"]): errors.append("invalid:messages")
    if len(canonical_content(record)) > MAX_CHARS: errors.append("invalid:too_large")
    if seen_ids is not None:
        if record["id"] in seen_ids: errors.append("duplicate:id")
        seen_ids.add(record["id"])
    return errors


def validate_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seen: set[str] = set(); valid: list[dict[str, Any]] = []; rejected: list[dict[str, Any]] = []
    for record in records:
        errors = validate_record(record, seen)
        copy = dict(record); copy["validation_status"] = "valid" if not errors else "rejected"; copy["verification"] = {**copy.get("verification", {}), "schema_errors": errors}
        (valid if not errors else rejected).append(copy)
    return valid, rejected


def deduplicate(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chosen: dict[str, dict[str, Any]] = {}; duplicates: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: item["id"]):
        digest = content_hash(record)
        if digest in chosen:
            duplicate = dict(record); duplicate["validation_status"] = "rejected"; duplicate["verification"] = {**duplicate.get("verification", {}), "duplicate_of": chosen[digest]["id"]}; duplicates.append(duplicate)
        else: chosen[digest] = record
    return list(chosen.values()), duplicates


def quality_score(record: dict[str, Any]) -> float:
    score = 0.35
    if record.get("validation_status") == "valid": score += 0.25
    if record.get("verification", {}).get("passed") is True: score += 0.25
    if record.get("reasoning") or record.get("trajectory"): score += 0.10
    if record.get("source", {}).get("url"): score += 0.05
    return round(min(score, 1.0), 2)


def statistics(records: list[dict[str, Any]], duplicate_count: int = 0, rejected_count: int = 0) -> dict[str, Any]:
    chars = [len(canonical_content(record)) for record in records]
    quality = [record.get("quality_score") for record in records if record.get("quality_score") is not None]
    return {"examples": len(records), "domains": dict(sorted(Counter(record["domain"] for record in records).items())), "sources": dict(sorted(Counter(record["source"]["name"] for record in records).items())), "characters": {"total": sum(chars), "mean": round(sum(chars) / len(chars), 2) if chars else 0, "max": max(chars, default=0)}, "quality": {"mean": round(sum(quality) / len(quality), 2) if quality else 0, "scored": len(quality)}, "duplicate_rate": duplicate_count / (len(records) + duplicate_count) if records or duplicate_count else 0, "validation_failure_rate": rejected_count / (len(records) + rejected_count) if records or rejected_count else 0}


def mix_records(records: list[dict[str, Any]], weights: dict[str, float], seed: int) -> list[dict[str, Any]]:
    if not weights or abs(sum(weights.values()) - 1.0) > 1e-9 or any(weight < 0 for weight in weights.values()):
        raise ValueError("Trọng số trộn phải không âm và có tổng bằng 1.0")
    groups = {domain: [record for record in records if record["domain"] == domain] for domain in weights}
    total = min((len(groups[domain]) / weight for domain, weight in weights.items() if weight), default=0)
    rng = random.Random(seed); selected: list[dict[str, Any]] = []
    for domain, weight in sorted(weights.items()):
        group = sorted(groups[domain], key=lambda item: item["id"]); rng.shuffle(group); selected.extend(group[:round(total * weight)])
    rng.shuffle(selected); return selected


def prepare_format(records: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    if kind == "sft": return [{"id": r["id"], "messages": sft_messages(r)} for r in records]
    if kind == "preference": return [{"id": r["id"], "prompt": r["input"], "chosen": r["expected_output"], "rejected": r["metadata"]["rejected_output"]} for r in records if r.get("metadata", {}).get("rejected_output")]
    if kind == "tool_use": return [r for r in records if r.get("tools_used")]
    if kind == "trajectory": return [r for r in records if r.get("trajectory")]
    if kind == "reasoning": return [r for r in records if r.get("reasoning")]
    raise ValueError(f"Không hỗ trợ định dạng huấn luyện: {kind}")


def verify_math(record: dict[str, Any]) -> dict[str, Any]:
    operators = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b}
    def calculate(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)): return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in operators: return operators[type(node.op)](calculate(node.left), calculate(node.right))
        raise ValueError("chỉ hỗ trợ phép tính số học + - * /")
    try:
        actual = str(calculate(ast.parse(record["input"], mode="eval").body))
        return {"kind": "math", "passed": actual == record["expected_output"], "actual": actual}
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError) as error: return {"kind": "math", "passed": False, "error": str(error)}


def verify_tool_call(record: dict[str, Any]) -> dict[str, Any]:
    trajectory = record.get("trajectory")
    passed = isinstance(trajectory, list) and all(isinstance(step, dict) and isinstance(step.get("tool"), str) and isinstance(step.get("arguments", {}), dict) for step in trajectory)
    return {"kind": "tool_call", "passed": passed}


def verify_python(record: dict[str, Any]) -> dict[str, Any]:
    try:
        compile(record["expected_output"], "<dataset-example>", "exec")
        return {"kind": "python_compile", "passed": True}
    except (SyntaxError, TypeError, ValueError) as error: return {"kind": "python_compile", "passed": False, "error": str(error)}


def generate_synthetic(teacher: TeacherModel, prompts: list[str], parser: Callable[[str], dict[str, Any]], verifier: Callable[[dict[str, Any]], dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []; rejected: list[dict[str, Any]] = []
    for output in teacher.generate_batch(prompts):
        try:
            record = parser(output); verification = verifier(record); record["verification"] = verification; record["quality_score"] = quality_score(record)
            (accepted if verification.get("passed") else rejected).append(record)
        except (ValueError, KeyError, json.JSONDecodeError) as error: rejected.append({"raw_output": output, "validation_status": "rejected", "verification": {"passed": False, "error": str(error)}})
    return accepted, rejected


def build_dataset(sources: list[str | Path], output_dir: str | Path, version: str, config: dict[str, Any], eval_sources: list[str | Path] | None = None, quality: Any = None) -> dict[str, Any]:
    """Kiểm tra schema → loại trùng tuyệt đối → bộ lọc chất lượng (nếu truyền `quality`, kiểu QualityConfig) → chặn trùng eval → xuất file và manifest."""
    raw = [record for source in sources for record in load_records(source)]
    valid, rejected = validate_records(raw); unique, duplicates = deduplicate(valid)
    quality_report: dict[str, Any] = {"enabled": False}
    if quality is not None:
        from local_ai.data.quality import apply_quality_filters  # quality.py dùng hàm của file này, nên import ở đây để tránh vòng lặp import
        unique, filtered, quality_report = apply_quality_filters(unique, quality); rejected = rejected + filtered; quality_report = {"enabled": True, **quality_report}
    overlap = find_eval_overlap(unique, [record for source in (eval_sources or []) for record in load_records(source)])
    if overlap: raise ValueError(f"Dữ liệu train trùng với dữ liệu eval (overlap): {', '.join(f'{identifier} (trùng {reason})' for identifier, reason in sorted(overlap))}")
    for record in unique: record["quality_score"] = quality_score(record)
    if config.get("mix_weights"): unique = mix_records(unique, config["mix_weights"], config.get("seed", 0))
    output = Path(output_dir); write_jsonl(output / "train.jsonl", unique); write_jsonl(output / "rejected.jsonl", rejected + duplicates)
    for kind in config.get("formats", ["sft"]): write_jsonl(output / f"{kind}.jsonl", prepare_format(unique, kind))
    checksum = hashlib.sha256((output / "train.jsonl").read_bytes()).hexdigest()
    manifest = {"version": version, "generated_at": datetime.now(timezone.utc).isoformat(), "source_manifest": [str(path) for path in sources], "configuration": config, "checksum": checksum, "statistics": statistics(unique, len(duplicates), len(rejected)), "quality": quality_report, "eval_sources": [str(path) for path in (eval_sources or [])]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
