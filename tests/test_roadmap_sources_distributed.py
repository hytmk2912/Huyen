import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_ai.data.corpus import Source, build_one, check_source_licenses, hf_dataset_id

CORPUS = "configs/datasets/corpus_10t.json"


def hf_source(**changes):
    base = {"source_id": "hf-vi", "dataset_name": "Nguồn thử", "dataset_version": "main", "url": "hf://datasets/test-org/test-vi", "license": "CC-BY-4.0", "license_url": "https://example.test/license", "domain": "vietnamese", "language": "vi", "estimated_size": 1, "estimated_tokens": 1, "download_method": "hf_dataset", "processing_recipe": "plain_text_v1", "options": {"text_field": "content", "max_rows": 2}}
    return Source.from_dict({**base, **changes})


class HfSourceAdapterTests(unittest.TestCase):
    def test_hf_dataset_source_builds_a_vietnamese_shard(self):
        calls = []
        def loader(name, subset=None, **kwargs):
            calls.append({"name": name, **kwargs})
            return iter([{"content": "Đoạn văn bản tiếng Việt thứ nhất dùng để kiểm thử bộ chuyển đổi nguồn dữ liệu."}, {"content": "Đoạn thứ hai có đủ ký tự khác nhau để vượt qua bộ lọc chất lượng."}, {"content": "Dòng thứ ba bị cắt bởi max_rows."}])
        config = json.loads(Path(CORPUS).read_text())
        with tempfile.TemporaryDirectory() as directory:
            config.update({"storage_root": directory, "tokenizer": {"kind": "utf8_bytes", "name": "test"}, "upload": {"enabled": False}})
            result = build_one(hf_source(), config, loader=loader)
        self.assertEqual(calls[0]["name"], "test-org/test-vi")
        self.assertTrue(calls[0]["streaming"])
        self.assertEqual(result["domain"], "vietnamese")
        self.assertGreater(result["token_count"], 0)

    def test_source_options_are_optional_and_urls_validated(self):
        config = json.loads(Path(CORPUS).read_text())
        self.assertEqual(Source.from_dict(config["sources"][0]).options, {})
        self.assertEqual(hf_dataset_id("hf://datasets/org/name"), "org/name")
        with self.assertRaisesRegex(ValueError, "hf://datasets"): hf_dataset_id("https://example.test/data")

    def test_unreviewed_license_is_rejected(self):
        config = json.loads(Path(CORPUS).read_text())
        check_source_licenses([hf_source()], config)
        with self.assertRaisesRegex(ValueError, "chưa được duyệt"): check_source_licenses([hf_source(license="Proprietary")], config)
        check_source_licenses([hf_source(license="Proprietary")], {})


class DistributedTrainingTests(unittest.TestCase):
    def test_fsdp_config_uses_bf16_and_torchrun(self):
        from local_ai.training.finetune import launch_command, load_finetune_config, resolve_base_model, training_arguments, training_dtype
        config = load_finetune_config("configs/training/sft_fsdp.json")
        self.assertEqual(training_dtype(config, resolve_base_model(config)), "bfloat16")
        args = training_arguments(config, "bfloat16")
        self.assertTrue(args["bf16"]); self.assertFalse(args["fp16"])
        self.assertEqual(args["fsdp"], "full_shard auto_wrap")
        self.assertTrue(args["fsdp_config"]["activation_checkpointing"])
        self.assertEqual(launch_command(config, "cfg.json")[:3], ["torchrun", "--nproc_per_node=8", "--nnodes=1"])

    def test_single_process_keeps_model_dtype(self):
        from local_ai.training.finetune import launch_command, load_finetune_config, resolve_base_model, training_arguments, training_dtype
        config = load_finetune_config("configs/training/sft.json")
        self.assertEqual(training_dtype(config, resolve_base_model(config)), "float32")
        self.assertNotIn("fsdp", training_arguments(config, "float32"))
        self.assertEqual(launch_command(config, "cfg.json")[0], "python")

    def test_invalid_distributed_settings_are_rejected(self):
        from local_ai.training.finetune import DistributedSettings, load_finetune_config
        with self.assertRaisesRegex(ValueError, "bfloat16"): load_finetune_config("configs/training/sft_fsdp.json", train_dtype="float32")
        with self.assertRaisesRegex(ValueError, "FP8"): load_finetune_config("configs/training/sft.json", train_dtype="fp8")
        with self.assertRaisesRegex(ValueError, "strategy"): load_finetune_config("configs/training/sft.json", distributed=DistributedSettings("zero"))

    def test_skips_when_fewer_gpus_than_processes(self):
        from local_ai.training import finetune
        config = finetune.load_finetune_config("configs/training/sft_fsdp.json")
        with mock.patch.object(finetune, "missing_requirements", return_value=["gpus>=8"]):
            self.assertEqual(finetune.train(config)["status"], "skipped")


if __name__ == "__main__": unittest.main()
