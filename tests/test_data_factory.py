import json
import sys
import types
import tempfile
import unittest
from pathlib import Path

from local_ai.data.core import (build_dataset, deduplicate, generate_synthetic, load_records, mix_records, prepare_format, statistics, validate_records, write_jsonl)


def record(identifier="one", domain="coding", content="same"):
    return {"id": identifier, "domain": domain, "task": "task", "input": content, "expected_output": "answer", "source": {"name": "test", "url": "https://example.test"}, "license": {"name": "CC0-1.0"}, "dataset_version": "v1", "difficulty": 1}


class FakeTeacher:
    def generate(self, prompt): return self.generate_batch([prompt])[0]
    def generate_batch(self, prompts): return [json.dumps(record(f"synthetic-{index}", "math_logic", "1 + 1")) for index, _ in enumerate(prompts)]


class DataFactoryTests(unittest.TestCase):
    def test_schema_validation_rejects_missing_invalid_and_duplicate_ids(self):
        invalid = record(); invalid.pop("license")
        valid, rejected = validate_records([record(), record(), invalid])
        self.assertEqual(len(valid), 1); self.assertEqual(len(rejected), 2)
        self.assertIn("duplicate:id", rejected[0]["verification"]["schema_errors"])

    def test_loading_json_jsonl_csv_and_deduplication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); records = [record("a"), record("b")]
            write_jsonl(root / "records.jsonl", records)
            (root / "records.json").write_text(json.dumps(records))
            (root / "records.csv").write_text("id,domain,task,input,expected_output,source,license,dataset_version\na,coding,t,x,y,source,license,v1\n")
            self.assertEqual(len(load_records(root / "records.jsonl")), 2)
            self.assertEqual(len(load_records(root / "records.json")), 2)
            self.assertEqual(len(load_records(root / "records.csv")), 1)
            unique, duplicates = deduplicate(records)
            self.assertEqual((len(unique), len(duplicates)), (1, 1))

    def test_mixing_stats_and_training_formats_are_deterministic(self):
        records = [record("a", "coding", "a"), record("b", "coding", "b"), record("c", "reasoning", "c"), record("d", "reasoning", "d")]
        mixed = mix_records(records, {"coding": 0.5, "reasoning": 0.5}, 3)
        self.assertEqual(mixed, mix_records(records, {"coding": 0.5, "reasoning": 0.5}, 3))
        self.assertEqual(statistics(records)["examples"], 4)
        self.assertEqual(len(prepare_format(records, "sft")), 4)

    def test_build_versions_and_prevents_eval_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "source.jsonl"; evaluation = root / "eval.jsonl"; write_jsonl(source, [record("train")]); write_jsonl(evaluation, [record("eval")])
            manifest = build_dataset([source], root / "out", "v2", {"seed": 1, "formats": ["sft"]}, [evaluation])
            self.assertEqual(manifest["version"], "v2"); self.assertTrue(manifest["checksum"]); self.assertTrue((root / "out" / "sft.jsonl").exists())
            write_jsonl(evaluation, [record("train")])
            with self.assertRaisesRegex(ValueError, "overlap"): build_dataset([source], root / "blocked", "v2", {}, [evaluation])

    def test_teacher_generation_requires_verification(self):
        accepted, rejected = generate_synthetic(FakeTeacher(), ["one", "two"], json.loads, lambda r: {"passed": r["input"] == "1 + 1"})
        self.assertEqual((len(accepted), len(rejected)), (2, 0))
        accepted, rejected = generate_synthetic(FakeTeacher(), ["one"], json.loads, lambda _: {"passed": False})
        self.assertEqual((len(accepted), len(rejected)), (0, 1))


if __name__ == "__main__": unittest.main()

class CorpusTests(unittest.TestCase):
    def test_tokenizer_and_hf_credentials(self):
        from local_ai.data.corpus import HuggingFaceUploader, Tokenizer
        self.assertEqual(Tokenizer({"kind": "utf8_bytes", "name": "test"}).encode("é"), [195, 169])
        import os
        old = os.environ.pop("HF_TOKEN", None)
        try:
            with self.assertRaisesRegex(RuntimeError, "HF_TOKEN"):
                HuggingFaceUploader("org/dataset", True)
        finally:
            if old: os.environ["HF_TOKEN"] = old

    def test_registry_progress_does_not_count_local_validation_as_upload(self):
        from local_ai.data.corpus import Registry
        with tempfile.TemporaryDirectory() as directory:
            registry = Registry(Path(directory))
            registry.shard(("s", "v", "local", "checksum", 9, 1, "code", "VALIDATED", "remote", "now"))
            progress = registry.progress(100)
            self.assertEqual(progress["approved_tokens"], 9)
            self.assertEqual(progress["uploaded_tokens"], 0)

    def test_production_tokenizer_is_deterministic(self):
        try: import tiktoken
        except ImportError: self.skipTest("optional tiktoken smoke dependency is unavailable")
        from local_ai.data.corpus import Tokenizer
        tokenizer = Tokenizer({"kind": "tiktoken", "name": "cl100k_base", "encoding": "cl100k_base", "revision": "tiktoken-0.14.0"})
        self.assertEqual(tokenizer.encode("deterministic token test"), tokenizer.encode("deterministic token test"))
        self.assertEqual(tokenizer.info()["revision"], "tiktoken-0.14.0")


class UploadTests(unittest.TestCase):
    def test_hf_upload_is_remote_verified_and_idempotent(self):
        from local_ai.data.corpus import HuggingFaceUploader
        calls = {"upload": 0}; remote = set()
        class Api:
            def __init__(self, token): self.token = token
            def file_exists(self, repo, path, repo_type): return path in remote
            def upload_file(self, **kwargs): calls["upload"] += 1; remote.add(kwargs["path_in_repo"])
        old_module, old_token = sys.modules.get("huggingface_hub"), __import__("os").environ.get("HF_TOKEN")
        sys.modules["huggingface_hub"] = types.SimpleNamespace(HfApi=Api); __import__("os").environ["HF_TOKEN"] = "test-token"
        try:
            with tempfile.TemporaryDirectory() as directory:
                shard = Path(directory) / "shard.jsonl"; shard.write_text("data")
                uploader = HuggingFaceUploader("org/test", True)
                self.assertEqual(uploader.upload_verify(shard, "train/a.jsonl", "checksum"), "REMOTE_VERIFIED")
                self.assertEqual(uploader.upload_verify(shard, "train/a.jsonl", "checksum"), "REMOTE_VERIFIED")
                self.assertEqual(calls["upload"], 1)
        finally:
            if old_module is None: sys.modules.pop("huggingface_hub", None)
            else: sys.modules["huggingface_hub"] = old_module
            if old_token is None: __import__("os").environ.pop("HF_TOKEN", None)
            else: __import__("os").environ["HF_TOKEN"] = old_token

    def test_hf_upload_failure_is_retryable(self):
        from local_ai.data.corpus import HuggingFaceUploader
        class Api:
            def __init__(self, token): pass
            def file_exists(self, *args, **kwargs): return False
            def upload_file(self, **kwargs): raise OSError("network down")
        old_module, old_token = sys.modules.get("huggingface_hub"), __import__("os").environ.get("HF_TOKEN")
        sys.modules["huggingface_hub"] = types.SimpleNamespace(HfApi=Api); __import__("os").environ["HF_TOKEN"] = "test-token"
        try:
            with tempfile.TemporaryDirectory() as directory:
                shard = Path(directory) / "shard.jsonl"; shard.write_text("data")
                with self.assertRaises(OSError): HuggingFaceUploader("org/test", True, retries=1).upload_verify(shard, "train/a.jsonl", "checksum")
                self.assertTrue(shard.exists())
        finally:
            if old_module is None: sys.modules.pop("huggingface_hub", None)
            else: sys.modules["huggingface_hub"] = old_module
            if old_token is None: __import__("os").environ.pop("HF_TOKEN", None)
            else: __import__("os").environ["HF_TOKEN"] = old_token
