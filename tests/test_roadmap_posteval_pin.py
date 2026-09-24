import json
import shutil
import tempfile
import types
import unittest
from pathlib import Path

from local_ai.data.corpus import Source, Tokenizer, measure_source, pin_source_revisions
from local_ai.training.finetune import evaluate_checkpoint, load_finetune_config

CORPUS = "configs/datasets/corpus_10t.json"
SHA = "b" * 40


class PostTrainingEvaluationTests(unittest.TestCase):
    def test_report_is_written_next_to_checkpoint(self):
        from local_ai.evaluation.suites import load_cases
        answers = {case.prompt: case.expected for case in load_cases("data/eval/vi_trading_eval.jsonl")}
        with tempfile.TemporaryDirectory() as directory:
            report = evaluate_checkpoint("data/eval/vi_trading_eval.jsonl", lambda prompt: answers[prompt], Path(directory) / "adapter")
            saved = json.loads((Path(directory) / "adapter" / "eval_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["accuracy"], 1.0)
        self.assertEqual(set(saved["suites"]), {"trading", "vietnamese"})

    def test_training_configs_enable_evaluation(self):
        for path in ("configs/training/sft.json", "configs/training/sft_fsdp.json"):
            config = load_finetune_config(path)
            self.assertTrue(Path(config.eval_cases).exists())
        self.assertIsNone(load_finetune_config("configs/training/sft.json", eval_cases="").eval_cases or None)


class PinAndMeasureSourcesTests(unittest.TestCase):
    def test_pin_only_unpinned_hf_sources(self):
        calls = []
        class Api:
            def dataset_info(self, repo, revision): calls.append((repo, revision)); return types.SimpleNamespace(sha=SHA)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.json"; raw = json.loads(Path(CORPUS).read_text())
            raw["sources"][-1]["dataset_version"] = "main"; path.write_text(json.dumps(raw))
            pinned = pin_source_revisions(path, Api())
            saved = json.loads(path.read_text())
        self.assertEqual([item["source_id"] for item in pinned], [raw["sources"][-1]["source_id"]])
        self.assertEqual(saved["sources"][-1]["dataset_version"], SHA)
        self.assertEqual(saved["sources"][0]["dataset_version"], "11-0")

    def test_shipped_hf_sources_are_pinned(self):
        for item in json.loads(Path(CORPUS).read_text())["sources"]:
            if item["download_method"] == "hf_dataset": self.assertRegex(item["dataset_version"], r"^[0-9a-f]{40}$")

    def test_measure_source_on_sample(self):
        config = json.loads(Path(CORPUS).read_text())
        source = Source.from_dict({**config["sources"][-2], "options": {"text_field": "text", "max_rows": 1000}})
        loader = lambda name, subset=None, **kwargs: iter([{"text": "abcd"}, {"text": "abcdef"}, {"text": ""}])
        result = measure_source(source, config, 10, loader, Tokenizer({"kind": "utf8_bytes", "name": "test"}))
        self.assertEqual((result["sample_rows"], result["avg_tokens_per_row"], result["estimated_tokens"]), (2, 5.0, 5000))
        http = Source.from_dict(config["sources"][0])
        self.assertEqual(measure_source(http, config)["status"], "skipped")


if __name__ == "__main__": unittest.main()


class TextTemplateTests(unittest.TestCase):
    def test_template_joins_columns_and_missing_column_is_empty(self):
        from local_ai.data.corpus import row_text
        self.assertEqual(row_text({"problem": "Đề", "solution": "Giải"}, {"text_template": "{problem}\n\n{solution}"}), "Đề\n\nGiải")
        self.assertEqual(row_text({"problem": "Đề"}, {"text_template": "{problem} {solution}"}), "")
        self.assertEqual(row_text({"content": " x "}, {"text_field": "content"}), "x")

    def test_reasoning_source_is_declared_and_pinned(self):
        sources = {item["source_id"]: item for item in json.loads(Path(CORPUS).read_text())["sources"]}
        self.assertEqual(sources["openr1-math"]["domain"], "reasoning")
        self.assertRegex(sources["openr1-math"]["dataset_version"], r"^[0-9a-f]{40}$")
