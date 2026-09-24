import json
import tempfile
import unittest
from pathlib import Path

from local_ai.config.settings import find_model_config, load_model_configs
from local_ai.data.manifest import DatasetManifest
from local_ai.experiments.tracking import RunTracker, seed_everything
from local_ai.models.adapters import ModelCapability, ModelConfig, quantization_settings
from local_ai.training.plans import TrainingPlan

PLATFORM = "configs/models/platform.json"


class ModelConfigTests(unittest.TestCase):
    def test_platform_lists_primary_and_three_huihui_qwen3_models_in_bf16(self):
        configs = load_model_configs(PLATFORM)
        self.assertEqual(
            [c.name for c in configs], ["primary", "huihui-qwen3-4b", "huihui-qwen3-8b", "huihui-qwen3-14b"]
        )
        self.assertEqual(configs[0].source, "huihui-ai/Huihui-Qwen3.8-27B-abliterated")
        self.assertIn(ModelCapability.CODING, configs[0].capabilities)
        for config in configs:
            self.assertEqual(config.dtype, "bfloat16")
            self.assertTrue(config.params_billion and config.configuration_hash)
        raw = json.loads(Path(PLATFORM).read_text())
        self.assertFalse(any("gguf_file" in item for item in raw["models"]))

    def test_lookup_duplicates_and_quantization_validation(self):
        self.assertEqual(find_model_config(PLATFORM, "huihui-qwen3-14b").params_billion, 14.8)
        with self.assertRaises(LookupError):
            find_model_config(PLATFORM, "missing")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.json"
            model = {"name": "same", "source": "org/model", "capabilities": ["chat"]}
            path.write_text(json.dumps({"models": [model, model]}))
            with self.assertRaisesRegex(ValueError, "trùng"):
                load_model_configs(path)
        self.assertEqual(
            ModelConfig.from_dict({"name": "q", "source": "o/m", "quantization": "nf4"}).quantization, "nf4"
        )
        with self.assertRaisesRegex(ValueError, "quantization"):
            ModelConfig.from_dict({"name": "q", "source": "o/m", "quantization": "int3"})
        self.assertEqual(quantization_settings("nf4")["bnb_4bit_quant_type"], "nf4")


class ReproducibilityTests(unittest.TestCase):
    def test_manifest_tracker_and_training_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "dataset.json"
            DatasetManifest("demo", "1", "local", "train").save(manifest)
            self.assertEqual(json.loads(manifest.read_text())["name"], "demo")
            tracker = RunTracker(Path(directory) / "run", {"seed": 1})
            tracker.record_metrics({"loss": 1.0})
            tracker.record_checkpoint("checkpoint-1")
            self.assertTrue((Path(directory) / "run" / "metrics.json").exists())
        seed_everything(12)
        TrainingPlan("sft", "sft.jsonl", "base", 12, "out").validate()
        with self.assertRaises(ValueError):
            TrainingPlan("sft", "", "base", 1, "out").validate()


if __name__ == "__main__":
    unittest.main()
