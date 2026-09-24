from __future__ import annotations

import ast
import csv
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from local_ai.data.schema import VALID_DOMAINS, VALID_STATUSES

REQUIRED = ("id", "domain", "task", "input", "expected_output", "source", "license", "dataset_version")
MAX_CHARS = 100_000


class TeacherModel(Protocol):
    def generate(self, prompt: str) -> str: ...
    def generate_batch(self, prompts: list[str]) -> list[str]: ...


def canonical_content(record: dict[str, Any]) -> str:
    fields = {
        key: record.get(key, "")
        for key in ("domain", "task", "input", "context", "expected_output", "reasoning", "trajectory", "messages")
    }
    return json.dumps(fields, sort_keys=True, separators=(",", ":"))


def content_hash(record: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_content(record).encode()).hexdigest()


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
            raise RuntimeError(
                "Đọc file parquet cần thư viện tùy chọn pyarrow; hãy cài nó để nạp file parquet"
            ) from error
        return pq.read_table(path).to_pylist()
    raise ValueError(f"Không hỗ trợ định dạng dữ liệu: {path.suffix}")


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")


MESSAGE_ROLES = {"system", "user", "assistant"}


def valid_messages(messages: Any) -> bool:
    """Hội thoại hợp lệ: mỗi lượt có role (system/user/assistant) và content không rỗng, có ít nhất một lượt user, lượt cuối là assistant."""
    if not isinstance(messages, list) or not messages:
        return False
    if not all(
        isinstance(m, dict)
        and m.get("role") in MESSAGE_ROLES
        and isinstance(m.get("content"), str)
        and m["content"].strip()
        for m in messages
    ):
        return False
    return any(m["role"] == "user" for m in messages) and messages[-1]["role"] == "assistant"


def sft_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    """Hội thoại cho sft.jsonl. Giữ nguyên messages nếu có; nếu không thì dựng từ input/expected_output,
    đưa context vào trước câu hỏi và reasoning vào khối <think>...</think> (định dạng suy luận của Qwen3)."""
    if record.get("messages"):
        return [{"role": m["role"], "content": m["content"]} for m in record["messages"]]
    question = f"{record['context']}\n\n{record['input']}" if record.get("context") else record["input"]
    answer = (
        f"<think>\n{record['reasoning']}\n</think>\n\n{record['expected_output']}"
        if record.get("reasoning")
        else record["expected_output"]
    )
    return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]


def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").casefold().split())


def eval_fingerprints(eval_records: list[dict[str, Any]]) -> tuple[set[str], set[str], set[str]]:
    """Dấu vân tay của tập eval: ID, content_hash và câu hỏi đã chuẩn hóa (trường input hoặc prompt)."""
    ids = {str(r["id"]) for r in eval_records if r.get("id")}
    hashes = {content_hash(r) for r in eval_records}
    questions = {normalize_text(r.get("input") or r.get("prompt")) for r in eval_records} - {""}
    return ids, hashes, questions


def split_eval_overlap(
    records: list[dict[str, Any]], eval_records: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Loại bản ghi train trùng với eval (cùng ID, cùng content_hash hoặc cùng câu hỏi), giữ lại phần còn lại."""
    ids, hashes, questions = eval_fingerprints(eval_records)
    kept: list[dict[str, Any]] = []
    leaked: list[dict[str, Any]] = []
    for record in records:
        user_turns = [m.get("content") for m in record.get("messages") or [] if m.get("role") == "user"]
        asked = {normalize_text(text) for text in [record.get("input"), *user_turns]} - {""}
        reason = (
            "id"
            if record["id"] in ids
            else "content_hash"
            if content_hash(record) in hashes
            else "question"
            if asked & questions
            else None
        )
        if reason is None:
            kept.append(record)
            continue
        copy = dict(record)
        copy["validation_status"] = "rejected"
        copy["verification"] = {**copy.get("verification", {}), "eval_overlap": reason}
        leaked.append(copy)
    return kept, leaked


def validate_record(record: dict[str, Any], seen_ids: set[str] | None = None) -> list[str]:
    errors = [f"missing:{key}" for key in REQUIRED if key not in record or record[key] in (None, "")]
    if errors:
        return errors
    if not isinstance(record["source"], dict) or not record["source"].get("name"):
        errors.append("invalid:source")
    if not isinstance(record["license"], dict) or not record["license"].get("name"):
        errors.append("invalid:license")
    if record["domain"] not in VALID_DOMAINS:
        errors.append("invalid:domain")
    if record.get("validation_status", "pending") not in VALID_STATUSES:
        errors.append("invalid:validation_status")
    if not isinstance(record.get("difficulty", 1), int) or not 1 <= record.get("difficulty", 1) <= 5:
        errors.append("invalid:difficulty")
    if record.get("quality_score") is not None and not 0 <= record["quality_score"] <= 1:
        errors.append("invalid:quality_score")
    if record.get("messages") is not None and not valid_messages(record["messages"]):
        errors.append("invalid:messages")
    if len(canonical_content(record)) > MAX_CHARS:
        errors.append("invalid:too_large")
    if seen_ids is not None:
        if record["id"] in seen_ids:
            errors.append("duplicate:id")
        seen_ids.add(record["id"])
    return errors


def validate_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seen: set[str] = set()
    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for record in records:
        errors = validate_record(record, seen)
        copy = dict(record)
        copy["validation_status"] = "valid" if not errors else "rejected"
        copy["verification"] = {**copy.get("verification", {}), "schema_errors": errors}
        (valid if not errors else rejected).append(copy)
    return valid, rejected


def deduplicate(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chosen: dict[str, dict[str, Any]] = {}
    duplicates: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: item["id"]):
        digest = content_hash(record)
        if digest in chosen:
            duplicate = dict(record)
            duplicate["validation_status"] = "rejected"
            duplicate["verification"] = {**duplicate.get("verification", {}), "duplicate_of": chosen[digest]["id"]}
            duplicates.append(duplicate)
        else:
            chosen[digest] = record
    return list(chosen.values()), duplicates


def quality_score(record: dict[str, Any]) -> float:
    score = 0.35
    if record.get("validation_status") == "valid":
        score += 0.25
    if record.get("verification", {}).get("passed") is True:
        score += 0.25
    if record.get("reasoning") or record.get("trajectory"):
        score += 0.10
    if record.get("source", {}).get("url"):
        score += 0.05
    return round(min(score, 1.0), 2)


def statistics(records: list[dict[str, Any]], duplicate_count: int = 0, rejected_count: int = 0) -> dict[str, Any]:
    chars = [len(canonical_content(record)) for record in records]
    quality = [record.get("quality_score") for record in records if record.get("quality_score") is not None]
    return {
        "examples": len(records),
        "domains": dict(sorted(Counter(record["domain"] for record in records).items())),
        "sources": dict(sorted(Counter(record["source"]["name"] for record in records).items())),
        "characters": {
            "total": sum(chars),
            "mean": round(sum(chars) / len(chars), 2) if chars else 0,
            "max": max(chars, default=0),
        },
        "quality": {"mean": round(sum(quality) / len(quality), 2) if quality else 0, "scored": len(quality)},
        "duplicate_rate": duplicate_count / (len(records) + duplicate_count) if records or duplicate_count else 0,
        "validation_failure_rate": rejected_count / (len(records) + rejected_count) if records or rejected_count else 0,
    }


def mix_records(records: list[dict[str, Any]], weights: dict[str, float], seed: int) -> list[dict[str, Any]]:
    if not weights or abs(sum(weights.values()) - 1.0) > 1e-9 or any(weight < 0 for weight in weights.values()):
        raise ValueError("Trọng số trộn phải không âm và có tổng bằng 1.0")
    groups = {domain: [record for record in records if record["domain"] == domain] for domain in weights}
    total = min((len(groups[domain]) / weight for domain, weight in weights.items() if weight), default=0)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for domain, weight in sorted(weights.items()):
        group = sorted(groups[domain], key=lambda item: item["id"])
        rng.shuffle(group)
        selected.extend(group[: round(total * weight)])
    rng.shuffle(selected)
    return selected


def prepare_format(records: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    if kind == "sft":
        return [{"id": r["id"], "messages": sft_messages(r)} for r in records]
    if kind == "preference":
        return [
            {
                "id": r["id"],
                "prompt": r["input"],
                "chosen": r["expected_output"],
                "rejected": r["metadata"]["rejected_output"],
            }
            for r in records
            if r.get("metadata", {}).get("rejected_output")
        ]
    if kind == "tool_use":
        return [r for r in records if r.get("tools_used")]
    if kind == "trajectory":
        return [r for r in records if r.get("trajectory")]
    if kind == "reasoning":
        return [r for r in records if r.get("reasoning")]
    raise ValueError(f"Không hỗ trợ định dạng huấn luyện: {kind}")


def verify_math(record: dict[str, Any]) -> dict[str, Any]:
    operators = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
    }

    def calculate(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            return operators[type(node.op)](calculate(node.left), calculate(node.right))
        raise ValueError("unsupported arithmetic")

    try:
        actual = str(calculate(ast.parse(record["input"], mode="eval").body))
        return {"kind": "math", "passed": actual == record["expected_output"], "actual": actual}
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError) as error:
        return {"kind": "math", "passed": False, "error": str(error)}


def verify_tool_call(record: dict[str, Any]) -> dict[str, Any]:
    trajectory = record.get("trajectory")
    passed = isinstance(trajectory, list) and all(
        isinstance(step, dict) and isinstance(step.get("tool"), str) and isinstance(step.get("arguments", {}), dict)
        for step in trajectory
    )
    return {"kind": "tool_call", "passed": passed}


def verify_python(record: dict[str, Any]) -> dict[str, Any]:
    try:
        compile(record["expected_output"], "<dataset-example>", "exec")
        return {"kind": "python_compile", "passed": True}
    except (SyntaxError, TypeError, ValueError) as error:
        return {"kind": "python_compile", "passed": False, "error": str(error)}


def generate_synthetic(
    teacher: TeacherModel,
    prompts: list[str],
    parser: Callable[[str], dict[str, Any]],
    verifier: Callable[[dict[str, Any]], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for output in teacher.generate_batch(prompts):
        try:
            record = parser(output)
            verification = verifier(record)
            record["verification"] = verification
            record["quality_score"] = quality_score(record)
            (accepted if verification.get("passed") else rejected).append(record)
        except (ValueError, KeyError, json.JSONDecodeError) as error:
            rejected.append(
                {
                    "raw_output": output,
                    "validation_status": "rejected",
                    "verification": {"passed": False, "error": str(error)},
                }
            )
    return accepted, rejected


def build_dataset(
    sources: list[str | Path],
    output_dir: str | Path,
    version: str,
    config: dict[str, Any],
    eval_sources: list[str | Path] | None = None,
) -> dict[str, Any]:
    raw = [record for source in sources for record in load_records(source)]
    valid, rejected = validate_records(raw)
    unique, duplicates = deduplicate(valid)
    unique, leaked = split_eval_overlap(
        unique, [record for source in (eval_sources or []) for record in load_records(source)]
    )
    for record in unique:
        record["quality_score"] = quality_score(record)
    if config.get("mix_weights"):
        unique = mix_records(unique, config["mix_weights"], config.get("seed", 0))
    output = Path(output_dir)
    write_jsonl(output / "train.jsonl", unique)
    write_jsonl(output / "rejected.jsonl", rejected + duplicates + leaked)
    for kind in config.get("formats", ["sft"]):
        write_jsonl(output / f"{kind}.jsonl", prepare_format(unique, kind))
    checksum = hashlib.sha256((output / "train.jsonl").read_bytes()).hexdigest()
    manifest = {
        "version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": [str(path) for path in sources],
        "configuration": config,
        "checksum": checksum,
        "statistics": {**statistics(unique, len(duplicates), len(rejected)), "eval_overlap_removed": len(leaked)},
        "eval_sources": [str(path) for path in (eval_sources or [])],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
