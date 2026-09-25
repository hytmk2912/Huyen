"""So sánh 2 báo cáo eval (ví dụ trước và sau khi fine-tune): `python -m local_ai.evaluation.compare TRUOC.json SAU.json`."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _rate(bucket: dict[str, Any] | None) -> str:
    return "—" if not bucket else f"{bucket['passed']}/{bucket['cases']} ({bucket['passed'] / bucket['cases']:.0%})"


def _change(before: dict[str, Any] | None, after: dict[str, Any] | None) -> str:
    if not before or not after: return "—"
    points = round(100 * (after["passed"] / after["cases"] - before["passed"] / before["cases"]))
    return f"{points:+d} điểm %"


def compare_reports(before: dict[str, Any], after: dict[str, Any]) -> str:
    """Bảng Markdown: tổng, theo nhóm, theo ngôn ngữ; kèm danh sách câu mới đạt và câu mới trượt."""
    if before.get("status") != "completed" or after.get("status") != "completed": raise ValueError('Cả hai báo cáo phải có status "completed"')
    total = lambda report: {"passed": report["passed"], "cases": report["cases"]}
    rows = [("Tổng", total(before), total(after))]
    for key, label in (("groups", "nhóm"), ("languages", "ngôn ngữ")):
        for name in sorted(set(before[key]) | set(after[key])): rows.append((f"{label} {name}", before[key].get(name), after[key].get(name)))
    lines = [f"So sánh eval: {before['model']} (trước) → {after['model']} (sau)", "", "| Phần | Trước | Sau | Thay đổi |", "| --- | ---: | ---: | ---: |"]
    lines += [f"| {name} | {_rate(old)} | {_rate(new)} | {_change(old, new)} |" for name, old, new in rows]
    failed_before = {item["id"] for item in before["failures"]}; failed_after = {item["id"] for item in after["failures"]}
    lines += ["", f"Câu mới đạt: {', '.join(sorted(failed_before - failed_after)) or 'không có'}", f"Câu mới trượt: {', '.join(sorted(failed_after - failed_before)) or 'không có'}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m local_ai.evaluation.compare", description="So sánh 2 báo cáo eval (report.json) và in bảng thay đổi.")
    parser.add_argument("before", help="File report.json của lần chấm trước khi train")
    parser.add_argument("after", help="File report.json của lần chấm sau khi train")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ kiểm tra tham số, không cần file báo cáo đã có")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"status": "dry-run", "before": args.before, "after": args.after}, ensure_ascii=False)); return 0
    missing = [path for path in (args.before, args.after) if not Path(path).is_file()]
    if missing:
        print(f"Không tìm thấy báo cáo: {', '.join(missing)}; hãy chạy eval trước", file=sys.stderr); return 2
    print(compare_reports(*(json.loads(Path(path).read_text(encoding="utf-8")) for path in (args.before, args.after))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
