"""Bộ đánh giá theo nhóm (ví dụ tiếng Việt, trading): nạp câu hỏi, chấm điểm, báo cáo theo từng nhóm."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

Scoring = Literal["exact", "contains", "numeric"]
NUMBER = re.compile(r"-?\d+(?:[.,]\d+)*")


@dataclass(frozen=True)
class EvalCase:
    id: str
    suite: str
    prompt: str
    expected: str
    scoring: Scoring = "exact"
    tolerance: float = 0.0


def load_cases(path: str | Path) -> list[EvalCase]:
    cases = [
        EvalCase(**json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Mã câu hỏi bị trùng trong {path}")
    for case in cases:
        if case.scoring not in ("exact", "contains", "numeric"):
            raise ValueError(f"Câu {case.id}: không hỗ trợ cách chấm {case.scoring}")
    return cases


def normalize(text: str) -> str:
    """Chuẩn hóa Unicode (NFC) và khoảng trắng, bỏ phân biệt hoa thường; giữ nguyên dấu tiếng Việt."""
    return " ".join(unicodedata.normalize("NFC", text).casefold().split()).strip(" .")


def parse_number(text: str) -> float | None:
    """Lấy số đầu tiên; hiểu cả kiểu Việt Nam (1.234,5) lẫn kiểu Anh (1,234.5)."""
    match = NUMBER.search(text.replace(" ", ""))
    if not match:
        return None
    raw = match.group()
    if "," in raw and "." in raw:
        raw = raw.replace(".", "").replace(",", ".") if raw.rfind(",") > raw.rfind(".") else raw.replace(",", "")
    elif raw.count(",") > 1 or raw.count(".") > 1:
        raw = raw.replace(",", "").replace(".", "")
    elif "," in raw or "." in raw:
        # Một dấu phân cách với đúng 3 chữ số phía sau (ví dụ 50.000) được hiểu là phân cách hàng nghìn.
        head, tail = re.split(r"[.,]", raw)
        raw = head + tail if len(tail) == 3 and head.lstrip("-") != "0" else f"{head}.{tail}"
    return float(raw)


def score(case: EvalCase, output: str) -> bool:
    if case.scoring == "exact":
        return normalize(output) == normalize(case.expected)
    if case.scoring == "contains":
        return normalize(case.expected) in normalize(output)
    actual, expected = parse_number(output), parse_number(case.expected)
    return actual is not None and expected is not None and abs(actual - expected) <= case.tolerance


def run_suites(cases: list[EvalCase], runner: Callable[[str], str]) -> dict[str, object]:
    suites: dict[str, dict[str, float]] = {}
    failures: list[str] = []
    for case in cases:
        try:
            passed = score(case, runner(case.prompt))
        except Exception:
            passed = False
        bucket = suites.setdefault(case.suite, {"cases": 0, "correct": 0})
        bucket["cases"] += 1
        bucket["correct"] += int(passed)
        if not passed:
            failures.append(case.id)
    for bucket in suites.values():
        bucket["accuracy"] = bucket["correct"] / bucket["cases"]
    total = sum(bucket["cases"] for bucket in suites.values())
    return {
        "cases": total,
        "accuracy": sum(bucket["correct"] for bucket in suites.values()) / total if total else 0.0,
        "suites": dict(sorted(suites.items())),
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m local_ai.evaluation.suites", description="Chạy bộ đánh giá theo nhóm cho một model"
    )
    parser.add_argument("--cases", default="data/eval/vi_trading_eval.jsonl", help="File câu hỏi JSONL")
    parser.add_argument("--model", default="primary", help="Tên model trong danh sách model")
    parser.add_argument("--models-config", default="configs/models/platform.json", help="File danh sách model")
    parser.add_argument("--report", help="Ghi báo cáo JSON ra file này")
    args = parser.parse_args(argv)
    import importlib.util

    from local_ai.config.settings import find_model_config
    from local_ai.contracts import Message
    from local_ai.models.adapters import HuggingFaceModelAdapter

    cases = load_cases(args.cases)
    missing = [name for name in ("torch", "transformers") if importlib.util.find_spec(name) is None]
    if missing:
        report = {"status": "skipped", "missing": missing, "cases": len(cases)}
    else:
        adapter = HuggingFaceModelAdapter(find_model_config(args.models_config, args.model))
        report = {
            "status": "completed",
            "model": args.model,
            **run_suites(cases, lambda prompt: adapter.generate([Message("user", prompt)])),
        }
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
