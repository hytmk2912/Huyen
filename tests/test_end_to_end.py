"""Kiểm thử đầu-cuối toàn bộ đường ống bằng dữ liệu giả, không cần mạng hay GPU:
build-corpus → verify-shard → hf-sft → finetune --dry-run → đánh giá theo nhóm."""
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from local_ai.data.core import load_records
from local_ai.data.corpus import Registry, Source, build_one, domain_mixture, verify_one_shard
from local_ai.data.hub import prepare_hf_sft
from local_ai.evaluation.suites import load_cases
from local_ai.training.finetune import describe, evaluate_checkpoint, load_finetune_config

CORPUS_ROWS = [{"problem": f"Bài toán số {i}: tính tổng của {i} và {i + 1} rồi giải thích từng bước.", "solution": f"Ta cộng {i} với {i + 1} được {2 * i + 1}. Kết quả cuối cùng là {2 * i + 1}, đã kiểm tra lại bằng phép trừ."} for i in range(5)]
SFT_ROWS = [{"messages": [{"role": "user", "content": f"Câu hỏi {i}"}, {"role": "assistant", "content": f"Trả lời {i}"}]} for i in range(3)]


class EndToEndPipelineTest(unittest.TestCase):
    def test_full_pipeline_with_fake_data(self):
        remote = set()
        class Api:
            def __init__(self, token): pass
            def file_exists(self, repo, path, repo_type): return path in remote
            def upload_file(self, **kwargs): remote.add(kwargs["path_in_repo"])
        corpus = json.loads(Path("configs/datasets/corpus_10t.json").read_text())
        reasoning = next(item for item in corpus["sources"] if item["source_id"] == "openr1-math")
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(HfApi=Api)}), \
                mock.patch.dict(os.environ, {"HF_TOKEN": "test-token", "HF_DATASET_REPO": "org/corpus"}):
            root = Path(directory)
            # 1. build-corpus: nguồn reasoning thật trong cấu hình, dữ liệu giả, tokenizer byte
            config = {**corpus, "storage_root": str(root / "storage"), "tokenizer": {"kind": "utf8_bytes", "name": "test"}, "upload": {"enabled": False}}
            shard = build_one(Source.from_dict(reasoning), config, loader=lambda *a, **k: iter(CORPUS_ROWS))
            self.assertEqual((shard["domain"], shard["validation_status"]), ("reasoning", "VALIDATED"))
            self.assertIn("Kết quả cuối cùng", (root / "storage" / "raw" / "openr1-math.txt").read_text(encoding="utf-8"))
            # 2. verify-shard: tải lên và xác minh trên kho Hugging Face giả
            verified = verify_one_shard({**config, "upload": {"retries": 1}})
            self.assertEqual(verified["status"], "REMOTE_VERIFIED")
            progress = Registry(root / "storage").progress(corpus["target_tokens"], domain_mixture(corpus))
            self.assertEqual(progress["by_domain_target"]["reasoning"]["verified_tokens"], shard["token_count"])
            # 3. hf-sft: dataset hội thoại giả → sft.jsonl
            spec = {"name": "test-org/sft", "license": {"name": "CC0-1.0"}, "messages_field": "messages", "domain": "reasoning"}
            prepare_hf_sft({"dataset_version": "e2e", "hf_dataset": spec}, root / "sft", lambda *a, **k: iter(SFT_ROWS))
            self.assertEqual(len(load_records(root / "sft" / "sft.jsonl")), 3)
            # 4. finetune --dry-run: kế hoạch trỏ đúng sft.jsonl và model gốc
            plan = describe(load_finetune_config("configs/training/sft.json", dataset_path=str(root / "sft" / "sft.jsonl"), output_dir=str(root / "run")))
            self.assertEqual(plan["base_model"]["name"], "huihui-qwen3-4b")
            self.assertTrue(Path(plan["dataset_path"]).exists())
            # 5. đánh giá theo nhóm: model giả trả lời đúng mọi câu, báo cáo nằm cạnh checkpoint
            answers = {case.prompt: case.expected for case in load_cases("data/eval/vi_trading_eval.jsonl")}
            report = evaluate_checkpoint("data/eval/vi_trading_eval.jsonl", answers.get, root / "run" / "adapter")
            self.assertEqual(report["accuracy"], 1.0)
            self.assertTrue((root / "run" / "adapter" / "eval_report.json").exists())
        self.assertEqual(len(remote), 1)


if __name__ == "__main__": unittest.main()
