"""Test của phần corpus 10T đã cất vào archive (không còn chạy trong tests/).
Để chạy lại cần đưa corpus.py về local_ai/data/ như trước."""
import sys
import tempfile
import types
import unittest
from pathlib import Path


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
