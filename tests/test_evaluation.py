import tempfile
import unittest
from pathlib import Path

from local_ai.evaluation.suites import EvalCase, load_cases, parse_number, run_suites, score
from local_ai.training.finetune import evaluate_checkpoint, load_finetune_config

CASES = "data/eval/vi_trading_eval.jsonl"


class SuiteEvaluationTests(unittest.TestCase):
    def test_bundled_cases_cover_vietnamese_and_trading(self):
        cases = load_cases(CASES)
        self.assertEqual({case.suite for case in cases}, {"vietnamese", "trading"})
        answers = {case.prompt: case.expected for case in cases}
        report = run_suites(cases, lambda prompt: answers[prompt])
        self.assertEqual(report["accuracy"], 1.0)
        self.assertEqual(report["suites"]["trading"]["cases"], 4)

    def test_scoring_rules(self):
        self.assertTrue(score(EvalCase("a", "vietnamese", "p", "Hà Nội", "contains"), "Đáp án: hà nội."))
        self.assertTrue(score(EvalCase("b", "trading", "p", "1000000", "numeric"), "1.000.000 đồng"))
        self.assertTrue(score(EvalCase("c", "trading", "p", "12.5", "numeric", 0.01), "Khoảng 12,5%"))
        self.assertFalse(score(EvalCase("d", "trading", "p", "3", "numeric"), "không biết"))
        self.assertTrue(score(EvalCase("e", "vietnamese", "p", "lạnh", "exact"), " Lạnh. "))
        self.assertEqual(parse_number("0.125"), 0.125)

    def test_report_groups_failures_and_survives_runner_errors(self):
        cases = [EvalCase("v", "vietnamese", "p1", "a"), EvalCase("t", "trading", "p2", "5", "numeric")]

        def runner(prompt):
            if prompt == "p2":
                raise TimeoutError
            return "a"

        report = run_suites(cases, runner)
        self.assertEqual(report["failures"], ["t"])
        self.assertEqual(
            (report["suites"]["vietnamese"]["accuracy"], report["suites"]["trading"]["accuracy"]), (1.0, 0.0)
        )


class PostTrainingEvaluationTests(unittest.TestCase):
    def test_report_is_written_next_to_checkpoint(self):
        answers = {case.prompt: case.expected for case in load_cases(CASES)}
        with tempfile.TemporaryDirectory() as directory:
            report = evaluate_checkpoint(CASES, answers.get, Path(directory) / "adapter")
            self.assertTrue((Path(directory) / "adapter" / "eval_report.json").exists())
        self.assertEqual(report["accuracy"], 1.0)

    def test_training_configs_enable_evaluation(self):
        for path in ("configs/training/sft.json", "configs/training/sft_qlora.json"):
            self.assertTrue(Path(load_finetune_config(path).eval_cases).exists())


if __name__ == "__main__":
    unittest.main()
