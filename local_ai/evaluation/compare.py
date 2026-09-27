"""So sánh 2 báo cáo eval (ví dụ trước và sau khi fine-tune): `python -m local_ai.evaluation.compare TRUOC.json SAU.json`."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def _rate(bucket: dict[str, Any] | None) -> str:
    return "—" if not bucket else f"{bucket['passed']}/{bucket['cases']} ({bucket['passed'] / bucket['cases']:.0%})"


def _change(before: dict[str, Any] | None, after: dict[str, Any] | None) -> str:
    if not before or not after: return "—"
    points = round(100 * (after["passed"] / after["cases"] - before["passed"] / before["cases"]))
    return f"{points:+d} điểm %"


def mcnemar_exact_p(newly_failed: int, newly_passed: int) -> float:
    """p-value hai phía của kiểm định McNemar chính xác, chỉ dùng thư viện chuẩn.

    `newly_failed` (b): số câu trước đạt, sau trượt; `newly_passed` (c): số câu trước trượt, sau đạt.
    Câu giữ nguyên kết quả không ảnh hưởng. Với n = b + c, k = min(b, c): p = min(1, 2 · Σ_{i=0..k} C(n, i) · 0,5^n); n = 0 thì p = 1.
    Ví dụ tool_use 7/8 → 3/8 (4 câu mới trượt, 0 câu mới đạt): p = 0,125, chưa đủ chắc để nói là tụt thật.
    """
    if newly_failed < 0 or newly_passed < 0: raise ValueError("Số câu mới trượt và mới đạt không được âm")
    n = newly_failed + newly_passed
    if n == 0: return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(newly_failed, newly_passed) + 1)) / 2 ** n)


def _format_p(value: float) -> str:
    return f"{value:.3f}".replace(".", ",")


def _flips(before: dict[str, Any], after: dict[str, Any], field: str | None, name: str | None) -> tuple[int, int] | None:
    """(b, c) của một phần; None nếu không ghép cặp được (khác số câu, hoặc câu trượt không ghi nhóm/ngôn ngữ)."""
    if field is None: old_bucket, new_bucket = before, after
    else: old_bucket, new_bucket = before[f"{field}s"].get(name), after[f"{field}s"].get(name)
    if not old_bucket or not new_bucket or old_bucket["cases"] != new_bucket["cases"] or before["cases"] != after["cases"]: return None
    def failed(report: dict[str, Any]) -> set[str] | None:
        if field is None: return {item["id"] for item in report["failures"]}
        if any(field not in item for item in report["failures"]): return None
        return {item["id"] for item in report["failures"] if item[field] == name}
    old, new = failed(before), failed(after)
    if old is None or new is None: return None
    return len(new - old), len(old - new)


def significance_lines(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """Bảng p-value (McNemar chính xác) cho tổng, từng nhóm và từng ngôn ngữ."""
    parts: list[tuple[str, str | None, str | None]] = [("Tổng", None, None)]
    for key, field, label in (("groups", "group", "nhóm"), ("languages", "language", "ngôn ngữ")):
        parts += [(f"{label} {name}", field, name) for name in sorted(set(before[key]) | set(after[key]))]
    lines = ["", "Kiểm định McNemar chính xác: p < 0,05 mới coi là thật, không phải ngẫu nhiên.", "",
             "| Phần | Mới trượt | Mới đạt | p-value | Kết luận |", "| --- | ---: | ---: | ---: | --- |"]
    for label, field, name in parts:
        flips = _flips(before, after, field, name)
        if flips is None: lines.append(f"| {label} | — | — | — | không ghép cặp được câu hỏi |"); continue
        p = mcnemar_exact_p(*flips)
        verdict = "không đổi" if sum(flips) == 0 else ("thật (p < 0,05)" if p < 0.05 else "chưa chắc, có thể do ngẫu nhiên")
        lines.append(f"| {label} | {flips[0]} | {flips[1]} | {_format_p(p)} | {verdict} |")
    return lines


def compare_reports(before: dict[str, Any], after: dict[str, Any]) -> str:
    """Bảng Markdown: tổng, theo nhóm, theo ngôn ngữ; kèm danh sách câu mới đạt, câu mới trượt và p-value của từng phần."""
    if before.get("status") != "completed" or after.get("status") != "completed": raise ValueError('Cả hai báo cáo phải có status "completed"')
    total = lambda report: {"passed": report["passed"], "cases": report["cases"]}
    rows = [("Tổng", total(before), total(after))]
    for key, label in (("groups", "nhóm"), ("languages", "ngôn ngữ")):
        for name in sorted(set(before[key]) | set(after[key])): rows.append((f"{label} {name}", before[key].get(name), after[key].get(name)))
    lines = [f"So sánh eval: {before['model']} (trước) → {after['model']} (sau)", "", "| Phần | Trước | Sau | Thay đổi |", "| --- | ---: | ---: | ---: |"]
    lines += [f"| {name} | {_rate(old)} | {_rate(new)} | {_change(old, new)} |" for name, old, new in rows]
    failed_before = {item["id"] for item in before["failures"]}; failed_after = {item["id"] for item in after["failures"]}
    lines += ["", f"Câu mới đạt: {', '.join(sorted(failed_before - failed_after)) or 'không có'}", f"Câu mới trượt: {', '.join(sorted(failed_after - failed_before)) or 'không có'}"]
    lines += significance_lines(before, after)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m local_ai.evaluation.compare", description="So sánh 2 báo cáo eval (report.json) và in bảng thay đổi.")
    parser.add_argument("before", help="File report.json của lần chấm trước khi train")
    parser.add_argument("after", help="File report.json của lần chấm sau khi train")
    parser.add_argument("--output", help="Thư mục ghi report.md (bảng so sánh kèm p-value); bỏ trống thì chỉ in ra màn hình")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ kiểm tra tham số, không cần file báo cáo đã có")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"status": "dry-run", "before": args.before, "after": args.after, "output": args.output}, ensure_ascii=False)); return 0
    missing = [path for path in (args.before, args.after) if not Path(path).is_file()]
    if missing:
        print(f"Không tìm thấy báo cáo: {', '.join(missing)}; hãy chạy eval trước", file=sys.stderr); return 2
    table = compare_reports(*(json.loads(Path(path).read_text(encoding="utf-8")) for path in (args.before, args.after)))
    print(table)
    if args.output:
        directory = Path(args.output); directory.mkdir(parents=True, exist_ok=True)
        (directory / "report.md").write_text(f"# {table}\n", encoding="utf-8")
        print(f"Đã ghi {directory / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
