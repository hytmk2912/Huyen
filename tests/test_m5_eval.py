"""Mốc M5: đánh giá (eval) với 4 cách chấm, bộ ~30 câu Việt + Anh, báo cáo JSON/Markdown và chặn trùng train/eval."""
import contextlib
import io
import json
import tempfile
import time
import unicodedata
import unittest
from pathlib import Path

from local_ai.data.core import build_dataset, find_eval_overlap, write_jsonl
from local_ai.evaluation.__main__ import main as eval_main
from local_ai.evaluation.suite import EvalCase, extract_code, load_cases, markdown_report, run_eval, score, write_reports
from local_ai.models.router import ScriptedModelAdapter

ROOT = Path(__file__).resolve().parent.parent


def case(scoring, **fields):
    return EvalCase(id=f"t-{scoring}", language="vi", group="reasoning", prompt="câu hỏi", scoring=scoring, **fields)


class ScoringTests(unittest.TestCase):
    def test_exact(self):
        self.assertTrue(score(case("exact", expected="Canberra"), "  canberra. ")[0])
        self.assertTrue(score(case("exact", expected="42"), "<think>6 * 7 = 42, chắc chắn</think>\n42")[0])
        passed, reason = score(case("exact", expected="Canberra"), "Sydney")
        self.assertFalse(passed); self.assertIn("Canberra", reason)

    def test_contains_normalizes_case_space_and_vietnamese_unicode(self):
        expected = case("contains", expected="Hà Nội")
        self.assertTrue(score(expected, unicodedata.normalize("NFD", "Thủ đô là HÀ   NỘI."))[0])  # chữ tổ hợp, hoa, nhiều dấu cách
        self.assertFalse(score(expected, "Thủ đô là Hồ Chí Minh")[0])
        self.assertFalse(score(expected, "<think>Hà Nội?</think> Không biết")[0])  # đáp án chỉ nằm trong <think> thì không tính

    def test_regex(self):
        weekday = case("regex", pattern=r"thứ\s*(năm|5)\b")
        self.assertTrue(score(weekday, "Ba ngày nữa là Thứ Năm.")[0])
        self.assertTrue(score(weekday, unicodedata.normalize("NFD", "thứ năm"))[0])
        self.assertFalse(score(weekday, "Thứ Sáu")[0])

    def test_python_tests_pass_and_fail(self):
        add = case("python_tests", tests="assert add(2, 3) == 5")
        self.assertTrue(score(add, "Đây là code:\n```python\ndef add(a, b):\n    return a + b\n```")[0])
        passed, reason = score(add, "def add(a, b):\n    return a - b")
        self.assertFalse(passed); self.assertIn("AssertionError", reason)

    def test_python_syntax_runtime_error_and_infinite_loop_do_not_crash(self):
        self.assertIn("SyntaxError", score(case("python_tests", tests="assert add(1, 1) == 2"), "def add(a, b) return a + b")[1])
        self.assertIn("ZeroDivisionError", score(case("python_tests", tests="assert add(1, 1) == 2"), "def add(a, b):\n    return 1 / 0")[1])
        start = time.monotonic()
        passed, reason = score(case("python_tests", tests="assert loop() == 1", timeout_s=1), "def loop():\n    while True:\n        pass")
        self.assertFalse(passed); self.assertIn("lặp vô hạn", reason)
        self.assertLess(time.monotonic() - start, 10)

    def test_extract_code(self):
        self.assertEqual(extract_code("a\n```python\nx = 1\n```\nb\n```\ny = 2\n```"), "x = 1\n\n\ny = 2\n")
        self.assertEqual(extract_code("x = 1"), "x = 1")


class CaseFileTests(unittest.TestCase):
    def test_case_file_has_about_30_questions_in_both_languages(self):
        cases = load_cases()
        self.assertTrue(28 <= len(cases) <= 35)
        languages = [item.language for item in cases]
        self.assertGreaterEqual(languages.count("vi"), 10); self.assertGreaterEqual(languages.count("en"), 10)
        self.assertEqual({item.group for item in cases}, {"code", "reasoning", "tool_use"})
        self.assertGreaterEqual(sum(item.scoring == "python_tests" for item in cases), 5)
        self.assertEqual({item.scoring for item in cases}, {"exact", "contains", "regex", "python_tests"})

    def test_every_reference_answer_passes_its_own_scoring(self):
        for item in load_cases():
            with self.subTest(item.id):
                self.assertTrue(item.reference)
                self.assertEqual(score(item, item.reference), (True, ""))

    def test_invalid_cases_are_rejected(self):
        for fields, message in (({"scoring": "fuzzy"}, "scoring"), ({"scoring": "exact"}, "expected"), ({"scoring": "regex", "pattern": "("}, "unterminated|missing"), ({"scoring": "exact", "expected": "x", "language": "fr"}, "language")):
            with self.subTest(fields), self.assertRaisesRegex(Exception, message):
                EvalCase(**{"id": "x", "language": "vi", "group": "code", "prompt": "p", **fields})


class RunTests(unittest.TestCase):
    def test_scripted_model_with_reference_answers_scores_100(self):
        cases = load_cases()
        report = run_eval(ScriptedModelAdapter("mau", [item.reference for item in cases]), cases)
        self.assertEqual((report["status"], report["passed"], report["accuracy"]), ("completed", len(cases), 1.0))
        self.assertEqual(sum(bucket["cases"] for bucket in report["languages"].values()), len(cases))

    def test_report_by_group_and_language_lists_failures(self):
        cases = [case("exact", expected="1"), EvalCase(id="en-1", language="en", group="code", prompt="p", scoring="contains", expected="ok")]
        report = run_eval(ScriptedModelAdapter("m", ["2", "ok"]), cases)
        self.assertEqual(report["groups"], {"code": {"cases": 1, "passed": 1, "accuracy": 1.0}, "reasoning": {"cases": 1, "passed": 0, "accuracy": 0.0}})
        self.assertEqual(report["languages"]["vi"]["passed"], 0)
        self.assertEqual([item["id"] for item in report["failures"]], ["t-exact"])
        markdown = markdown_report(report)
        self.assertIn("## Câu không đạt", markdown); self.assertIn("`t-exact`", markdown); self.assertIn("| code | 1 | 1 | 100% |", markdown)

    def test_model_error_stops_without_crash(self):
        report = run_eval(ScriptedModelAdapter("het", []), [case("exact", expected="1")])
        self.assertEqual(report["status"], "error"); self.assertIn("t-exact", report["error"])
        self.assertIn("error", markdown_report(report))

    def test_reports_are_written_as_json_and_markdown(self):
        cases = [case("exact", expected="1")]
        with tempfile.TemporaryDirectory() as directory:
            json_path, markdown_path = write_reports(run_eval(ScriptedModelAdapter("m", ["1"]), cases), directory)
            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8"))["accuracy"], 1.0)
            self.assertIn("Báo cáo eval", markdown_path.read_text(encoding="utf-8"))


class CommandTests(unittest.TestCase):
    def run_command(self, argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr): code = eval_main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_scripted_run_writes_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            code, stdout, _ = self.run_command(["--scripted", "--output", directory])
            self.assertEqual(code, 0); self.assertIn("30/30", stdout)
            self.assertEqual(json.loads((Path(directory) / "report.json").read_text(encoding="utf-8"))["model"], "scripted-reference")
            self.assertTrue((Path(directory) / "report.md").exists())

    def test_eval_by_model_name_reports_server_error(self):
        with tempfile.TemporaryDirectory() as directory:
            models = Path(directory) / "models.json"
            models.write_text(json.dumps({"models": [{"name": "local", "backend": "openai_compatible", "base_url": "http://127.0.0.1:9/v1", "source": "x", "capabilities": ["chat"], "timeout_s": 2}]}), encoding="utf-8")
            code, _, stderr = self.run_command(["--model", "local", "--models", str(models), "--output", f"{directory}/out"])
            self.assertEqual(code, 1); self.assertIn("Model lỗi", stderr)
            self.assertEqual(json.loads((Path(directory) / "out" / "report.json").read_text(encoding="utf-8"))["status"], "error")

    def test_training_data_overlapping_eval_is_blocked(self):
        prompt = load_cases()[0].prompt
        with tempfile.TemporaryDirectory() as directory:
            sft = Path(directory) / "sft.jsonl"
            write_jsonl(sft, [{"id": "khac-id", "messages": [{"role": "user", "content": prompt.upper()}, {"role": "assistant", "content": "x"}]}])
            code, _, stderr = self.run_command(["--scripted", "--train-data", str(sft), "--output", f"{directory}/out"])
            self.assertEqual(code, 2); self.assertIn("khac-id (trùng câu hỏi)", stderr)
            self.assertFalse((Path(directory) / "out").exists())


class OverlapTests(unittest.TestCase):
    def record(self, identifier, question, answer="đáp án"):
        return {"id": identifier, "domain": "reasoning", "task": "t", "input": question, "expected_output": answer, "source": {"name": "s"}, "license": {"name": "CC0-1.0"}, "dataset_version": "v"}

    def test_overlap_by_id_content_and_question(self):
        eval_records = [self.record("e1", "Thủ đô của Việt Nam là gì?"), {"id": "e2", "prompt": "What is 17 * 23?"}]
        train = [self.record("e1", "khác hẳn"), self.record("t1", "  THỦ ĐÔ của Việt Nam là gì  "), {"id": "t2", "messages": [{"role": "user", "content": "what is 17 * 23"}]}, self.record("t3", "Câu hỏi riêng")]
        self.assertEqual(find_eval_overlap(train, eval_records), [("e1", "id"), ("t1", "câu hỏi"), ("t2", "câu hỏi")])
        self.assertEqual(find_eval_overlap([self.record("t9", "Thủ đô của Việt Nam là gì?")], [self.record("e9", "Thủ đô của Việt Nam là gì?")]), [("t9", "nội dung")])

    def test_build_blocks_same_question_with_different_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(root / "train.jsonl", [self.record("train-1", load_cases()[5].prompt)])
            with self.assertRaisesRegex(ValueError, "overlap.*train-1 \\(trùng câu hỏi\\)"):
                build_dataset([root / "train.jsonl"], root / "out", "v", {}, [ROOT / "data" / "eval" / "eval_v1.jsonl"])

    def test_dataset_configs_block_the_new_eval_set(self):
        for path in [ROOT / "configs" / "datasets" / "hf_sft.json", *sorted((ROOT / "configs" / "datasets" / "presets").glob("*.json"))]:
            with self.subTest(path.name):
                self.assertIn("data/eval/eval_v1.jsonl", json.loads(path.read_text(encoding="utf-8"))["eval_sources"])


class ReadmeTests(unittest.TestCase):
    def test_readme_explains_eval(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for phrase in ("python -m local_ai.evaluation --model", "--scripted", "python_tests", "regex", "contains", "exact", "--train-data", ".runs/eval"):
            self.assertIn(phrase, readme)


if __name__ == "__main__":
    unittest.main()
