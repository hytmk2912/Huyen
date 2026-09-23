import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from local_ai.data.corpus import Source, build_one, check_source_domains, check_source_licenses
from local_ai.data.dedup import NearDedupIndex, NearDedupSettings, estimated_jaccard, shingles, signature, _permutations

CORPUS = "configs/datasets/corpus_10t.json"
BASE = "Thị trường chứng khoán hôm nay biến động mạnh khi nhà đầu tư lo ngại lãi suất tăng và dòng tiền rút khỏi nhóm cổ phiếu ngân hàng trong phiên chiều"
OTHER = "Mô hình ngôn ngữ cần dữ liệu sạch, đa dạng và có giấy phép rõ ràng để huấn luyện ổn định mà không bị lặp lại nội dung quá nhiều lần giữa các nguồn khác nhau"


class NearDedupTests(unittest.TestCase):
    def setUp(self):
        self.index = NearDedupIndex(sqlite3.connect(":memory:"), NearDedupSettings(min_words=10))

    def test_similarity_estimate(self):
        perms = _permutations(128)
        same = estimated_jaccard(signature(shingles(BASE, 5), perms), signature(shingles(BASE + " thêm", 5), perms))
        different = estimated_jaccard(signature(shingles(BASE, 5), perms), signature(shingles(OTHER, 5), perms))
        self.assertGreater(same, 0.8); self.assertLess(different, 0.2)

    def test_drops_near_duplicates_across_and_within_sources(self):
        kept, stats = self.index.filter_text("a", f"{BASE}\n\n{OTHER}")
        self.assertEqual(stats["near_duplicates"], 0)
        kept, stats = self.index.filter_text("b", f"{BASE} thêm.\n\nNgắn thôi.\n\n{BASE}")
        self.assertEqual(stats["near_duplicates"], 2)
        self.assertEqual(kept, "Ngắn thôi.")
        self.assertTrue(stats["examples"][0]["duplicate_of"].startswith("a:"))

    def test_rebuilding_a_source_does_not_match_itself(self):
        self.index.filter_text("a", BASE)
        kept, stats = self.index.filter_text("a", BASE)
        self.assertEqual((kept, stats["near_duplicates"]), (BASE, 0))

    def test_settings_validation(self):
        with self.assertRaisesRegex(ValueError, "bands"): NearDedupSettings.from_config({"near_dedup": {"num_perm": 100, "bands": 32}})
        self.assertEqual(NearDedupSettings.from_config({}).threshold, 0.8)

    def test_build_rejects_source_that_is_entirely_duplicate(self):
        config = json.loads(Path(CORPUS).read_text())
        text = "\n\n".join([BASE, OTHER])
        def loader(name, subset=None, **kwargs): return iter([{"text": text}])
        def source(source_id): return Source.from_dict({**config["sources"][-2], "source_id": source_id, "options": {"text_field": "text"}})
        with tempfile.TemporaryDirectory() as directory:
            config.update({"storage_root": directory, "tokenizer": {"kind": "utf8_bytes", "name": "test"}, "upload": {"enabled": False}, "near_dedup": {"min_words": 10}})
            first = build_one(source("vi-a"), config, loader=loader)
            second = build_one(source("vi-b"), config, loader=loader)
        self.assertEqual(first["near_dedup"]["near_duplicates"], 0)
        self.assertEqual(second["rejected"], "near_duplicate")


class DeclaredSourcesTests(unittest.TestCase):
    def test_trading_and_vietnamese_have_licensed_sources(self):
        config = json.loads(Path(CORPUS).read_text())
        sources = [Source.from_dict(item) for item in config["sources"]]
        check_source_domains(sources, config); check_source_licenses(sources, config)
        domains = {source.domain for source in sources}
        self.assertTrue({"trading", "vietnamese", "general", "code"} <= domains)
        vietnamese = next(source for source in sources if source.domain == "vietnamese")
        self.assertEqual((vietnamese.download_method, vietnamese.options["subset"]), ("hf_dataset", "vie_Latn"))


if __name__ == "__main__": unittest.main()
