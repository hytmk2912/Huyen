"""So sánh 2 lần chạy agent trên cùng bộ nhiệm vụ (M18), ví dụ model gốc và model đã train.

`python -m local_ai.agents.compare .runs/agent/goc.json .runs/agent/da_train.json --labels "model gốc" "model đã train"`

Đầu vào là file JSON do `python -m local_ai.agents.tasks --output ...` ghi. In bảng từng nhiệm vụ (đạt hay không, lý do),
tỉ lệ thành công, thời gian, và các nhiệm vụ mới đạt / mới trượt ở lần chạy thứ hai. `--output` ghi kết quả ra JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_report(path: str | Path) -> dict[str, Any]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report.get("status") != "completed" or not isinstance(report.get("tasks"), list):
        raise ValueError(f"{path} không phải báo cáo agent đã chạy xong (cần status completed và danh sách tasks); hãy chạy python -m local_ai.agents.tasks --output ...")
    return report


def compare_reports(first: dict[str, Any], second: dict[str, Any], labels: tuple[str, str] = ("lần 1", "lần 2")) -> dict[str, Any]:
    """Ghép kết quả theo id nhiệm vụ (thứ tự của lần 1, nhiệm vụ chỉ có ở lần 2 xếp sau)."""
    before, after = ({item["id"]: item for item in report["tasks"]} for report in (first, second))
    ids = list(before) + [identifier for identifier in after if identifier not in before]
    rows = [{"id": identifier, "first": _task(before.get(identifier)), "second": _task(after.get(identifier))} for identifier in ids]
    common = [row for row in rows if row["first"] and row["second"]]
    return {"labels": list(labels), "models": [first.get("model"), second.get("model")], "tasks": rows,
            "summary": [_summary(report) for report in (first, second)],
            "newly_passed": [row["id"] for row in common if row["second"]["success"] and not row["first"]["success"]],
            "newly_failed": [row["id"] for row in common if row["first"]["success"] and not row["second"]["success"]],
            "only_in_one": [row["id"] for row in rows if not (row["first"] and row["second"])]}


def _task(item: dict[str, Any] | None) -> dict[str, Any] | None:
    if item is None: return None
    return {"success": bool(item.get("success")), "reason": item.get("reason"), "tools_used": item.get("tools_used", []), "duration_s": item.get("duration_s")}


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {"passed": report.get("passed"), "total": report.get("total"), "success_rate": report.get("success_rate"), "minutes": round((report.get("duration_s") or 0) / 60, 1)}


def _cell(task: dict[str, Any] | None) -> str:
    if task is None: return "—"
    return "ĐẠT" if task["success"] else f"không đạt ({task['reason']})"


def format_comparison(result: dict[str, Any]) -> str:
    first, second = result["labels"]
    lines = [f"So sánh agent: {first} ({result['models'][0]}) và {second} ({result['models'][1]})", "", f"| Nhiệm vụ | {first} | {second} |", "| --- | --- | --- |"]
    lines += [f"| {row['id']} | {_cell(row['first'])} | {_cell(row['second'])} |" for row in result["tasks"]]
    a, b = result["summary"]
    lines.append(f"| **Tỉ lệ thành công** | {a['passed']}/{a['total']} ({a['success_rate']:.0%}) | {b['passed']}/{b['total']} ({b['success_rate']:.0%}) |")
    lines.append(f"| Thời gian | {a['minutes']:.1f} phút | {b['minutes']:.1f} phút |".replace(".", ","))
    lines += ["", f"Mới đạt ở {second}: {', '.join(result['newly_passed']) or 'không có'}.", f"Mới trượt ở {second}: {', '.join(result['newly_failed']) or 'không có'}."]
    if result["only_in_one"]: lines.append(f"Chỉ có ở một lần chạy (không so): {', '.join(result['only_in_one'])}.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m local_ai.agents.compare", description="So sánh 2 báo cáo agent (JSON của python -m local_ai.agents.tasks --output) theo từng nhiệm vụ.")
    parser.add_argument("first", help="Báo cáo lần 1, ví dụ model gốc")
    parser.add_argument("second", help="Báo cáo lần 2, ví dụ model đã train")
    parser.add_argument("--labels", nargs=2, default=["lần 1", "lần 2"], metavar=("NHÃN_1", "NHÃN_2"), help="Tên hiển thị của 2 lần chạy")
    parser.add_argument("--output", help="Ghi kết quả so sánh (JSON) vào file này")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ in kế hoạch (2 file đầu vào có tồn tại không), không đọc báo cáo")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"status": "dry-run", "first": args.first, "second": args.second, "exists": [Path(args.first).is_file(), Path(args.second).is_file()], "labels": args.labels}, ensure_ascii=False, indent=2)); return 0
    result = compare_reports(load_report(args.first), load_report(args.second), tuple(args.labels))
    print(format_comparison(result))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
