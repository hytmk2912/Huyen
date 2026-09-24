import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_ai.config.settings import find_model_config
from local_ai.data.core import load_records
from local_ai.data.hub import load_hf_rows, map_rows, prepare_hf_sft

FIXTURE = Path(__file__).parent / "fixtures" / "hf_rows.jsonl"


def fixture_loader(calls):
    def load(name, subset=None, **kwargs):
        calls.append({"name": name, "subset": subset, **kwargs})
        return iter(load_records(FIXTURE))

    return load


def hf_config(**spec):
    base = {
        "name": "test-org/test-dataset",
        "split": "train",
        "license": {"name": "CC0-1.0"},
        "mapping": {"id": "uid", "input": "prompt", "expected_output": "response"},
    }
    return {"dataset_version": "hf-test", "seed": 1, "formats": ["sft"], "hf_dataset": {**base, **spec}}


class HuggingFaceDatasetStepTests(unittest.TestCase):
    def test_hf_sft_step_validates_deduplicates_and_exports(self):
        calls = []
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"HF_TOKEN": "env-token"}):
            manifest = prepare_hf_sft(hf_config(), directory, fixture_loader(calls))
            sft = load_records(Path(directory) / "sft.jsonl")
            rejected = load_records(Path(directory) / "rejected.jsonl")
        self.assertEqual(calls[0]["name"], "test-org/test-dataset")
        self.assertEqual(calls[0]["token"], "env-token")
        self.assertEqual(manifest["version"], "hf-test")
        self.assertEqual(len(sft), 2)
        self.assertEqual(sft[0]["messages"][0]["role"], "user")
        self.assertEqual(len(rejected), 3)

    def test_token_is_optional_and_limit_applies(self):
        calls = []
        with mock.patch.dict(os.environ, {}, clear=True):
            rows = load_hf_rows(hf_config(limit=2)["hf_dataset"], fixture_loader(calls))
        self.assertIsNone(calls[0]["token"])
        self.assertEqual(len(rows), 2)

    def test_messages_field_mapping(self):
        rows = load_records(FIXTURE)[4:]
        records = map_rows(rows, hf_config(messages_field="messages", mapping={})["hf_dataset"], "v")
        self.assertEqual(
            (records[0]["input"], records[0]["expected_output"]), ("fixture chat prompt", "fixture chat answer")
        )

    def test_spec_requires_dataset_name_and_license(self):
        with self.assertRaisesRegex(ValueError, "name"):
            prepare_hf_sft(hf_config(name=""), "unused", fixture_loader([]))
        with self.assertRaisesRegex(ValueError, "license"):
            prepare_hf_sft(hf_config(license={}), "unused", fixture_loader([]))

    def test_shipped_config_names_no_dataset(self):
        config = json.loads(Path("configs/datasets/hf_sft.json").read_text())
        self.assertEqual(config["hf_dataset"]["name"], "")


class FinetuneHarnessTests(unittest.TestCase):
    def test_config_resolves_base_model_and_overrides(self):
        from local_ai.training.finetune import describe, load_finetune_config

        config = load_finetune_config("configs/training/sft.json", method="full")
        self.assertEqual(config.method, "full")
        self.assertTrue(config.gradient_checkpointing)
        plan = describe(config)
        self.assertEqual(
            plan["base_model"]["source"], find_model_config(config.models_config, config.base_model).source
        )
        with self.assertRaisesRegex(ValueError, "method"):
            load_finetune_config("configs/training/sft.json", method="int8")

    def test_resume_picks_latest_checkpoint(self):
        from local_ai.training.finetune import load_finetune_config, resume_target

        with tempfile.TemporaryDirectory() as directory:
            for name in ("checkpoint-5", "checkpoint-40", "checkpoint-100", "adapter"):
                (Path(directory) / name).mkdir()
            config = load_finetune_config("configs/training/sft.json", output_dir=directory)
            self.assertEqual(Path(resume_target(config)).name, "checkpoint-100")
            self.assertIsNone(
                resume_target(load_finetune_config("configs/training/sft.json", output_dir=directory, resume=False))
            )
        self.assertIsNone(
            resume_target(load_finetune_config("configs/training/sft.json", output_dir=str(Path(directory) / "none")))
        )

    def test_train_skips_without_gpu_or_libraries(self):
        from local_ai.training import finetune

        config = finetune.load_finetune_config("configs/training/sft.json")
        with mock.patch.object(finetune, "missing_requirements", return_value=["cuda"]):
            self.assertEqual(finetune.train(config), {"status": "skipped", "missing": ["cuda"]})


class QLoRATests(unittest.TestCase):
    def test_qlora_plan_for_27b_fits_24gb_gpu(self):
        from local_ai.training.finetune import describe, load_finetune_config

        plan = describe(load_finetune_config("configs/training/sft_qlora.json"))
        self.assertEqual(
            (plan["method"], plan["weight_format"], plan["base_model"]["name"]), ("qlora", "nf4", "primary")
        )
        self.assertLess(plan["estimated_weight_memory_gb"], 24)
        bf16 = describe(load_finetune_config("configs/training/sft_qlora.json", method="lora"))
        self.assertEqual(bf16["estimated_weight_memory_gb"], 54.0)

    def test_qlora_requires_bitsandbytes_and_peft(self):
        from local_ai.training import finetune

        config = finetune.load_finetune_config("configs/training/sft_qlora.json", require_gpu=False)
        with mock.patch("importlib.util.find_spec", return_value=None):
            self.assertEqual(
                finetune.missing_requirements(config),
                ["torch", "transformers", "trl", "datasets", "peft", "bitsandbytes"],
            )
        self.assertTrue(finetune.training_arguments(config)["bf16"])


class EndToEndTests(unittest.TestCase):
    def test_hf_sft_then_dry_run_then_evaluation(self):
        from local_ai.evaluation.suites import load_cases
        from local_ai.training.finetune import describe, evaluate_checkpoint, load_finetune_config

        rows = [
            {
                "messages": [
                    {"role": "system", "content": "Trợ lý"},
                    {"role": "user", "content": f"Câu {i}"},
                    {"role": "assistant", "content": f"Đáp {i}"},
                ]
            }
            for i in range(3)
        ]
        spec = {"name": "test-org/chat", "license": {"name": "CC0-1.0"}, "messages_field": "messages"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_hf_sft({"dataset_version": "e2e", "hf_dataset": spec}, root / "sft", lambda *a, **k: iter(rows))
            sft = load_records(root / "sft" / "sft.jsonl")
            self.assertEqual(sft[0]["messages"][0]["role"], "system")
            plan = describe(
                load_finetune_config(
                    "configs/training/sft.json",
                    dataset_path=str(root / "sft" / "sft.jsonl"),
                    output_dir=str(root / "run"),
                )
            )
            self.assertTrue(Path(plan["dataset_path"]).exists())
            answers = {case.prompt: case.expected for case in load_cases("data/eval/vi_trading_eval.jsonl")}
            self.assertEqual(
                evaluate_checkpoint("data/eval/vi_trading_eval.jsonl", answers.get, root / "run" / "adapter")[
                    "accuracy"
                ],
                1.0,
            )


if __name__ == "__main__":
    unittest.main()
