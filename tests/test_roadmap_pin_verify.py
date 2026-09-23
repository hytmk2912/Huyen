import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from local_ai.config.settings import find_model_config
from local_ai.data.corpus import Registry, digest, verify_one_shard
from local_ai.models.smoke import is_pinned, pin_model, smoke_test

SHA = "a" * 40


class FakeApi:
    def __init__(self): self.calls = []
    def model_info(self, repo, revision):
        self.calls.append((repo, revision)); return types.SimpleNamespace(sha=SHA)


class FakeAdapter:
    def __init__(self, config): self.config = config
    def load(self): pass
    def generate(self, messages): return "5"
    def tokenizer_metadata(self): return {"tokenizer_id": self.config.source}


class PinAndSmokeTests(unittest.TestCase):
    def test_pin_writes_commit_for_model_and_tokenizer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "platform.json"; shutil.copy("configs/models/platform.json", path)
            api = FakeApi(); result = pin_model(path, "primary", api)
            self.assertEqual((result["revision"], result["tokenizer_revision"]), (SHA, SHA))
            self.assertTrue(is_pinned(find_model_config(path, "primary")))
            self.assertFalse(is_pinned(find_model_config(path, "huihui-qwen3-4b")))
            pin_model(path, "primary", api)
            self.assertEqual(len(api.calls), 2)
            with self.assertRaises(LookupError): pin_model(path, "missing", api)

    def test_smoke_test_reports_and_skips_without_libraries(self):
        config = find_model_config("configs/models/platform.json", "primary")
        with tempfile.TemporaryDirectory() as directory:
            report = smoke_test(config, report_dir=directory, adapter_factory=FakeAdapter)
            self.assertEqual(report["status"], "passed")
            self.assertTrue((Path(directory) / "primary.json").exists())
        with mock.patch("local_ai.models.smoke.missing_requirements", return_value=["cuda"]):
            self.assertEqual(smoke_test(config)["status"], "skipped")


class OneShardVerificationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(); root = Path(self.directory.name)
        shard = root / "shard.jsonl"; shard.write_text("data")
        Registry(root).shard(("s1", "v", str(shard), digest(shard), 10, 1, "code", "VALIDATED", "train/code/s1.jsonl", "now"))
        self.config = {"storage_root": str(root), "upload": {"repo": "org/test", "retries": 1}}
        self.remote = set()
        remote = self.remote
        class Api:
            def __init__(self, token): pass
            def file_exists(self, repo, path, repo_type): return path in remote
            def upload_file(self, **kwargs): remote.add(kwargs["path_in_repo"])
        self.patches = [mock.patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(HfApi=Api)}), mock.patch.dict(os.environ, {"HF_TOKEN": "test-token"})]
        for patch in self.patches: patch.start()

    def tearDown(self):
        for patch in self.patches: patch.stop()
        self.directory.cleanup()

    def test_verifies_one_shard_and_marks_complete(self):
        result = verify_one_shard(self.config)
        self.assertEqual((result["status"], result["shard_id"]), ("REMOTE_VERIFIED", "s1"))
        self.assertIn("train/code/s1.jsonl", self.remote)
        progress = Registry(Path(self.directory.name)).progress(100)
        self.assertEqual(progress["remotely_verified_tokens"], 10)
        self.assertEqual(verify_one_shard(self.config)["status"], "no_shard")

    def test_skips_without_token_and_rejects_bad_checksum(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(verify_one_shard({"upload": {}})["missing"], ["HF_TOKEN", "HF_DATASET_REPO"])
        (Path(self.directory.name) / "shard.jsonl").write_text("changed")
        with self.assertRaisesRegex(ValueError, "checksum"): verify_one_shard(self.config)


if __name__ == "__main__": unittest.main()
