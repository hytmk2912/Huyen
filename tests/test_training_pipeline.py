import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_ai.config.settings import find_model_config, load_model_configs
from local_ai.data.core import load_records
from local_ai.data.hub import load_hf_rows, map_rows, prepare_hf_sft

FIXTURE = Path(__file__).parent / "fixtures" / "hf_rows.jsonl"


def fixture_loader(calls):
    def load(name, subset=None, **kwargs):
        calls.append({"name": name, "subset": subset, **kwargs})
        return iter(load_records(FIXTURE))
    return load


def hf_config(**spec):
    base = {"name": "test-org/test-dataset", "split": "train", "license": {"name": "CC0-1.0"}, "mapping": {"id": "uid", "input": "prompt", "expected_output": "response"}}
    return {"dataset_version": "hf-test", "seed": 1, "formats": ["sft"], "hf_dataset": {**base, **spec}}


class MultiModelConfigTests(unittest.TestCase):
    def test_platform_config_lists_several_unique_models(self):
        configs = load_model_configs("configs/models/platform.json")
        self.assertGreaterEqual(len(configs), 3)
        self.assertEqual(configs[0].name, "primary")
        self.assertEqual(len({config.name for config in configs}), len(configs))
        for config in configs:
            self.assertTrue(config.source and config.dtype and config.capabilities)
        self.assertEqual(find_model_config("configs/models/platform.json", "smoke").name, "smoke")
        with self.assertRaises(LookupError): find_model_config("configs/models/platform.json", "missing")

    def test_duplicate_model_names_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.json"
            model = {"name": "same", "source": "org/model", "capabilities": ["chat"]}
            path.write_text(json.dumps({"models": [model, model]}))
            with self.assertRaisesRegex(ValueError, "trùng"): load_model_configs(path)


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
        self.assertEqual((records[0]["input"], records[0]["expected_output"]), ("fixture chat prompt", "fixture chat answer"))

    def test_spec_requires_dataset_name_and_license(self):
        with self.assertRaisesRegex(ValueError, "name"): prepare_hf_sft(hf_config(name=""), "unused", fixture_loader([]))
        with self.assertRaisesRegex(ValueError, "license"): prepare_hf_sft(hf_config(license={}), "unused", fixture_loader([]))

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
        self.assertEqual(plan["base_model"]["source"], find_model_config(config.models_config, config.base_model).source)
        with self.assertRaisesRegex(ValueError, "method"): load_finetune_config("configs/training/sft.json", method="qlora")

    def test_resume_picks_latest_checkpoint(self):
        from local_ai.training.finetune import load_finetune_config, resume_target
        with tempfile.TemporaryDirectory() as directory:
            for name in ("checkpoint-5", "checkpoint-40", "checkpoint-100", "adapter"): (Path(directory) / name).mkdir()
            config = load_finetune_config("configs/training/sft.json", output_dir=directory)
            self.assertEqual(Path(resume_target(config)).name, "checkpoint-100")
            self.assertIsNone(resume_target(load_finetune_config("configs/training/sft.json", output_dir=directory, resume=False)))
        self.assertIsNone(resume_target(load_finetune_config("configs/training/sft.json", output_dir=str(Path(directory) / "none"))))

    def test_train_skips_without_gpu_or_libraries(self):
        from local_ai.training import finetune
        config = finetune.load_finetune_config("configs/training/sft.json")
        with mock.patch.object(finetune, "missing_requirements", return_value=["cuda"]):
            self.assertEqual(finetune.train(config), {"status": "skipped", "missing": ["cuda"]})


if __name__ == "__main__": unittest.main()
