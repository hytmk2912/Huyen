"""M22: kiểm định ý nghĩa (McNemar chính xác) cho so sánh eval trước/sau.

- `mcnemar_exact_p` chỉ dùng thư viện chuẩn: p(0,4)=0,125; p(0,2)=0,5; p(0,6)≈0,0312; p(3,3)=1; n=0 → 1.
- `compare_reports` in p-value cho tổng, từng nhóm, từng ngôn ngữ, kèm chú thích "p < 0,05 mới coi là thật".
- Lệnh compare ghi `report.md` có các dòng p-value; `--dry-run` vẫn chạy.
"""
import ast
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from local_ai.evaluation import compare
from local_ai.evaluation.compare import compare_reports, main as compare_main, mcnemar_exact_p

NOTE = "p < 0,05 mới coi là thật, không phải ngẫu nhiên"


def case(case_id, group, language):
    return {"id": case_id, "group": group, "language": language}


def report(model, cases, failed_ids):
    """Báo cáo giống `suite.py`: đếm theo nhóm và ngôn ngữ, câu trượt ghi nhóm và ngôn ngữ."""
    def buckets(field):
        result = {}
        for item in cases:
            bucket = result.setdefault(item[field], {"passed": 0, "cases": 0})
            bucket["cases"] += 1; bucket["passed"] += item["id"] not in failed_ids
        return result
    failures = [item for item in cases if item["id"] in failed_ids]
    return {"status": "completed", "model": model, "cases": len(cases), "passed": len(cases) - len(failures),
            "groups": buckets("group"), "languages": buckets("language"), "failures": failures}


# 8 câu tool_use (7/8 → 3/8, như `light` đo 26/9) và 8 câu code không đổi kết quả.
CASES = [case(f"tool-{index}", "tool_use", "vi" if index < 4 else "en") for index in range(8)] + \
        [case(f"code-{index}", "code", "vi" if index < 4 else "en") for index in range(8)]
BEFORE = report("light", CASES, {"tool-0", "code-0"})
AFTER = report("light-colab", CASES, {"tool-0", "tool-1", "tool-2", "tool-5", "tool-6", "code-0"})


class McNemarExactTests(unittest.TestCase):
    def test_known_values(self):
        self.assertEqual(mcnemar_exact_p(0, 4), 0.125)
        self.assertEqual(mcnemar_exact_p(0, 2), 0.5)
        self.assertEqual(mcnemar_exact_p(0, 6), 0.03125); self.assertAlmostEqual(mcnemar_exact_p(0, 6), 0.0312, delta=1e-4)  # 2/64
        self.assertEqual(mcnemar_exact_p(3, 3), 1.0)
        self.assertEqual(mcnemar_exact_p(0, 0), 1.0)

    def test_symmetric_and_bounded(self):
        for b in range(12):
            for c in range(12):
                with self.subTest(b=b, c=c):
                    p = mcnemar_exact_p(b, c)
                    self.assertEqual(p, mcnemar_exact_p(c, b)); self.assertGreater(p, 0); self.assertLessEqual(p, 1)
        with self.assertRaisesRegex(ValueError, "không được âm"): mcnemar_exact_p(-1, 2)

    def test_standard_library_only(self):
        tree = ast.parse(Path(compare.__file__).read_text(encoding="utf-8"))
        imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        imported |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
        self.assertEqual(imported & {"scipy", "numpy", "statsmodels"}, set())
        self.assertIn("math", imported)


class CompareSignificanceTests(unittest.TestCase):
    def test_table_has_p_value_per_group_language_and_total(self):
        table = compare_reports(BEFORE, AFTER)
        self.assertIn(NOTE, table)
        self.assertIn("| nhóm tool_use | 7/8 (88%) | 3/8 (38%) | -50 điểm % |", table)  # bảng cũ giữ nguyên
        self.assertIn("| nhóm tool_use | 4 | 0 | 0,125 | chưa chắc, có thể do ngẫu nhiên |", table)
        self.assertIn("| nhóm code | 0 | 0 | 1,000 | không đổi |", table)
        self.assertIn("| Tổng | 4 | 0 | 0,125 | chưa chắc, có thể do ngẫu nhiên |", table)
        self.assertIn("| ngôn ngữ vi | 2 | 0 | 0,500 | chưa chắc, có thể do ngẫu nhiên |", table)
        self.assertIn("| ngôn ngữ en | 2 | 0 | 0,500 | chưa chắc, có thể do ngẫu nhiên |", table)

    def test_clear_change_is_marked_real(self):
        cases = [case(f"tool-{index}", "tool_use", "vi") for index in range(8)]
        table = compare_reports(report("a", cases, set()), report("b", cases, {f"tool-{index}" for index in range(6)}))
        self.assertIn("| nhóm tool_use | 6 | 0 | 0,031 | thật (p < 0,05) |", table)

    def test_unpaired_reports_show_dash(self):
        other = report("khac", CASES[:8], {"tool-0"})  # khác bộ câu: không ghép cặp được
        table = compare_reports(BEFORE, other)
        self.assertIn("| Tổng | — | — | — | không ghép cặp được câu hỏi |", table)
        self.assertIn("| nhóm code | — | — | — | không ghép cặp được câu hỏi |", table)


class CompareCommandTests(unittest.TestCase):
    def test_cli_prints_p_values_and_writes_report_md(self):
        with tempfile.TemporaryDirectory() as directory:
            before, after = Path(directory) / "truoc.json", Path(directory) / "sau.json"
            before.write_text(json.dumps(BEFORE), encoding="utf-8"); after.write_text(json.dumps(AFTER), encoding="utf-8")
            output_dir = Path(directory) / "so-sanh"
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                self.assertEqual(compare_main([str(before), str(after), "--output", str(output_dir)]), 0)
            self.assertIn(NOTE, printed.getvalue()); self.assertIn("0,125", printed.getvalue())
            markdown = (output_dir / "report.md").read_text(encoding="utf-8")
            self.assertTrue(markdown.startswith("# So sánh eval: light (trước) → light-colab (sau)"))
            self.assertIn(NOTE, markdown); self.assertIn("| nhóm tool_use | 4 | 0 | 0,125 |", markdown)

    def test_dry_run_still_works(self):
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            self.assertEqual(compare_main(["khong-co/truoc.json", "khong-co/sau.json", "--output", "khong-co/so-sanh", "--dry-run"]), 0)
        plan = json.loads(printed.getvalue())
        self.assertEqual(plan, {"status": "dry-run", "before": "khong-co/truoc.json", "after": "khong-co/sau.json", "output": "khong-co/so-sanh"})
        self.assertFalse(Path("khong-co").exists())


if __name__ == "__main__":
    unittest.main()
