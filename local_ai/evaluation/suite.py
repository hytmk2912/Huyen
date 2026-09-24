"""Bộ đánh giá: nạp câu hỏi, hỏi model, chấm điểm theo từng câu, báo cáo theo nhóm và ngôn ngữ.

Cách chấm (khóa `scoring` của mỗi câu):
- `exact`: câu trả lời bằng đúng `expected` sau khi chuẩn hóa (Unicode NFC, không phân biệt hoa thường, gộp khoảng trắng,
  bỏ dấu chấm câu ở hai đầu);
- `contains`: câu trả lời chứa `expected` sau khi chuẩn hóa;
- `regex`: câu trả lời khớp biểu thức `pattern` (không phân biệt hoa thường, `.` khớp cả xuống dòng);
- `python_tests`: lấy code trong câu trả lời (khối ```python nếu có), chạy cùng `tests` trong PythonSandbox có giới hạn thời gian;
  đạt khi chạy hết không lỗi. Code sai cú pháp, lỗi khi chạy hay lặp vô hạn đều tính là không đạt.
Khối <think>...</think> trong câu trả lời bị bỏ trước khi chấm.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from local_ai.contracts import Message, ModelAdapter
from local_ai.data.core import find_eval_overlap, load_records, normalize_text
from local_ai.tools.sandbox import PythonSandbox

SCORINGS = ("exact", "contains", "regex", "python_tests")
LANGUAGES = ("vi", "en")
GROUPS = ("code", "reasoning", "tool_use")
DEFAULT_CASES = Path(__file__).resolve().parents[2] / "data" / "eval" / "eval_v1.jsonl"
_THINK = re.compile(r"<think>.*?(</think>|$)", re.DOTALL | re.IGNORECASE)
_CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class EvalCase:
    id: str
    language: str
    group: str
    prompt: str
    scoring: str
    expected: str | None = None
    pattern: str | None = None
    tests: str | None = None
    timeout_s: float = 5.0
    reference: str | None = None  # một câu trả lời đúng mẫu, dùng để tự kiểm tra bộ câu hỏi

    def __post_init__(self) -> None:
        where = f"Câu eval '{self.id}'"
        if self.scoring not in SCORINGS: raise ValueError(f"{where}: scoring phải là một trong {', '.join(SCORINGS)}, không phải '{self.scoring}'")
        if self.language not in LANGUAGES: raise ValueError(f"{where}: language phải là {' hoặc '.join(LANGUAGES)}")
        if self.group not in GROUPS: raise ValueError(f"{where}: group phải là một trong {', '.join(GROUPS)}")
        required = {"exact": "expected", "contains": "expected", "regex": "pattern", "python_tests": "tests"}[self.scoring]
        if not getattr(self, required): raise ValueError(f"{where}: cách chấm {self.scoring} cần trường {required}")
        if self.pattern: re.compile(self.pattern)


def load_cases(path: str | Path = DEFAULT_CASES) -> list[EvalCase]:
    cases = [EvalCase(**record) for record in load_records(path)]
    ids = [case.id for case in cases]
    duplicates = sorted({identifier for identifier in ids if ids.count(identifier) > 1})
    if duplicates: raise ValueError(f"Mã câu eval bị trùng trong {path}: {', '.join(duplicates)}")
    return cases


def strip_think(output: str) -> str:
    return _THINK.sub("", output or "").strip()


def extract_code(output: str) -> str:
    """Code trong câu trả lời: nối các khối ```python ... ```; không có khối nào thì lấy cả câu trả lời."""
    blocks = _CODE_BLOCK.findall(output)
    return "\n\n".join(blocks) if blocks else output


def _normalized(text: str) -> str:
    return normalize_text(text).strip(" .!?")


def score(case: EvalCase, output: str, sandbox: PythonSandbox | None = None) -> tuple[bool, str]:
    """Chấm một câu trả lời. Trả về (đạt hay không, lý do ngắn khi không đạt)."""
    answer = strip_think(output)
    if case.scoring == "exact":
        passed = _normalized(answer) == _normalized(case.expected)
        return passed, "" if passed else f"cần đúng bằng '{case.expected}'"
    if case.scoring == "contains":
        passed = _normalized(case.expected) in _normalized(answer)
        return passed, "" if passed else f"không chứa '{case.expected}'"
    if case.scoring == "regex":
        passed = re.search(unicodedata.normalize("NFC", case.pattern), unicodedata.normalize("NFC", answer), re.IGNORECASE | re.DOTALL) is not None
        return passed, "" if passed else f"không khớp regex {case.pattern}"
    result = (sandbox or PythonSandbox()).run(extract_code(answer) + "\n\n" + case.tests + "\n", timeout_seconds=case.timeout_s)
    if result.timed_out: return False, f"quá {case.timeout_s:g} giây (có thể lặp vô hạn)"
    if result.returncode != 0:
        lines = [line for line in result.stderr.strip().splitlines() if line.strip()]
        return False, f"test không qua: {lines[-1] if lines else f'mã thoát {result.returncode}'}"
    return True, ""


def _bucket(buckets: dict[str, dict[str, Any]], key: str, passed: bool) -> None:
    bucket = buckets.setdefault(key, {"cases": 0, "passed": 0}); bucket["cases"] += 1; bucket["passed"] += int(passed)


def run_eval(model: ModelAdapter, cases: list[EvalCase], sandbox: PythonSandbox | None = None) -> dict[str, Any]:
    """Hỏi model từng câu và chấm. Model lỗi (ví dụ server chưa chạy) thì dừng và báo status "error", không chấm tiếp."""
    results, groups, languages = [], {}, {}
    for case in cases:
        try:
            output = model.generate([Message("user", case.prompt)])
        except Exception as error:  # lỗi hạ tầng, không phải câu trả lời sai
            return {"status": "error", "model": model.name, "error": f"Model lỗi ở câu {case.id}: {type(error).__name__}: {error}", "answered": len(results), "cases": len(cases)}
        passed, reason = score(case, output, sandbox)
        results.append({"id": case.id, "group": case.group, "language": case.language, "scoring": case.scoring, "passed": passed, "reason": reason, "output": output})
        _bucket(groups, case.group, passed); _bucket(languages, case.language, passed)
    for bucket in (*groups.values(), *languages.values()): bucket["accuracy"] = round(bucket["passed"] / bucket["cases"], 4)
    passed_total = sum(item["passed"] for item in results)
    return {"status": "completed", "model": model.name, "generated_at": datetime.now(timezone.utc).isoformat(), "cases": len(results), "passed": passed_total,
            "accuracy": round(passed_total / len(results), 4) if results else 0.0, "groups": dict(sorted(groups.items())), "languages": dict(sorted(languages.items())),
            "failures": [{key: item[key] for key in ("id", "group", "language", "scoring", "reason")} | {"output": item["output"][:300]} for item in results if not item["passed"]],
            "results": results}


def check_train_overlap(cases: list[EvalCase], train_paths: list[str | Path]) -> list[tuple[str, str]]:
    """Bản ghi train (train.jsonl hoặc sft.jsonl) trùng với câu eval theo id hoặc nội dung câu hỏi."""
    eval_records = [{"id": case.id, "prompt": case.prompt} for case in cases]
    return find_eval_overlap([record for path in train_paths for record in load_records(path)], eval_records)


def markdown_report(report: dict[str, Any]) -> str:
    if report["status"] != "completed": return f"# Báo cáo eval: {report['model']}\n\nTrạng thái: **{report['status']}**\n\n{report.get('error', '')}\n"
    def table(title: str, buckets: dict[str, dict[str, Any]]) -> list[str]:
        return [f"## Theo {title}", "", f"| {title.capitalize()} | Số câu | Đạt | Tỉ lệ |", "| --- | ---: | ---: | ---: |",
                *(f"| {name} | {bucket['cases']} | {bucket['passed']} | {bucket['accuracy']:.0%} |" for name, bucket in buckets.items()), ""]
    lines = [f"# Báo cáo eval: {report['model']}", "", f"Thời điểm: {report['generated_at']}", "", f"**Tổng: {report['passed']}/{report['cases']} câu đạt ({report['accuracy']:.0%})**", "",
             *table("nhóm", report["groups"]), *table("ngôn ngữ", report["languages"]), "## Câu không đạt", ""]
    lines += [f"- `{item['id']}` ({item['group']}, {item['language']}, {item['scoring']}): {item['reason']}" for item in report["failures"]] or ["Không có."]
    return "\n".join(lines) + "\n"


def write_reports(report: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    directory = Path(output_dir); directory.mkdir(parents=True, exist_ok=True)
    json_path, markdown_path = directory / "report.json", directory / "report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    return json_path, markdown_path
