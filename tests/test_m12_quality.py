"""Mốc M12: bộ lọc chất lượng dữ liệu (ngôn ngữ ưu tiên tiếng Việt, độ dài, lặp từ/câu, gần trùng bằng MinHash tự viết)
và thống kê trước/sau lọc trong manifest.json.

Mỗi bộ lọc có fixture riêng trong tests/fixtures/quality/; mỗi dòng ghi kết quả mong đợi ở `metadata.expected`
("keep" hoặc tên bộ lọc phải loại dòng đó). Không cần mạng hay thư viện ngoài.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

from local_ai.data.core import build_dataset, load_records
from local_ai.data.hub import load_preset, prepare_hf_mix
from local_ai.data.quality import (DEFAULT_QUALITY, LanguageFilter, MinHash, NearDuplicateFilter, QualityConfig, RepetitionFilter, apply_quality_filters,
                                   conversation, detect_language, duplicate_ngram_ratio, duplicate_sentence_ratio, estimated_jaccard, jaccard, longest_word_run, shingles)

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).parent / "fixtures" / "quality"
FILTERS = ("language", "length", "repetition", "near_duplicate")


def fixture(name):
    return load_records(FIXTURES / f"{name}.jsonl")


def expected(records):
    return {record["id"]: record["metadata"]["expected"] for record in records}


class FixtureCase(unittest.TestCase):
    config = QualityConfig.from_file()

    def assert_filter(self, name):
        """Chạy riêng một bộ lọc trên fixture của nó: dòng "keep" phải qua, dòng còn lại phải bị chính bộ lọc đó loại."""
        records = fixture(name)
        self.assertIn("keep", expected(records).values()); self.assertIn(name, expected(records).values())
        for record in records:
            with self.subTest(record["id"]):
                reason = getattr(self.config, name).check(record)
                self.assertEqual(name if reason else "keep", record["metadata"]["expected"], reason)
                if reason: self.assertRegex(reason, r"[à-ỹ]")  # lý do viết bằng tiếng Việt


class LanguageFilterTests(FixtureCase):
    def test_language_fixture(self):
        self.assert_filter("language")

    def test_detect_language(self):
        cases = {"vi": "Hôm nay trời đẹp, chúng ta cùng đi dạo công viên nhé.", "vi-khong-dau": "Hom nay troi dep, chung ta cung di dao cong vien nhe, ban co muon di khong?",
                 "en": "The quick brown fox jumps over the lazy dog and runs to the forest.", "other": "今天天气很好，我们一起去公园散步吧。这是一个很好的主意。",
                 "unknown": "x = 2 + 3 * 4"}
        for language, text in cases.items():
            with self.subTest(language): self.assertEqual(detect_language(text), language)

    def test_vietnamese_first_even_when_mixed_with_code_or_english(self):
        record = {r["id"]: r for r in fixture("language")}["lang-vi-code"]
        self.assertEqual(detect_language("\n".join(conversation(record))), "vi")
        self.assertEqual(detect_language("Hàm này trả về tổng của hai số. The function returns the sum of two numbers, and it is very simple to use in any program."), "vi")

    def test_allowed_languages_come_from_config(self):
        english = {r["id"]: r for r in fixture("language")}["lang-en"]
        self.assertIsNone(LanguageFilter().check(english))
        self.assertIn("không nằm trong danh sách giữ", LanguageFilter(allowed=("vi", "unknown")).check(english))
        self.assertIsNone(LanguageFilter(match_source_language=False).check({r["id"]: r for r in fixture("language")}["lang-en-trong-nguon-vi"]))
        vietnamese_in_code = {r["id"]: r for r in fixture("language")}["lang-vi-trong-nguon-en"]
        self.assertEqual((vietnamese_in_code["metadata"]["language"], LanguageFilter().check(vietnamese_in_code)), ("en", None))  # ưu tiên tiếng Việt


class LengthFilterTests(FixtureCase):
    def test_length_fixture(self):
        self.assert_filter("length")


class RepetitionFilterTests(FixtureCase):
    def test_repetition_fixture(self):
        self.assert_filter("repetition")

    def test_measures(self):
        self.assertEqual(longest_word_run("tôi rất rất rất vui".split()), 3)
        self.assertEqual(longest_word_run("0 0 0 0 0 x x x a".split()), 0)  # số và ký hiệu một chữ không tính
        self.assertEqual(longest_word_run("1 cm 2 cm 3 cm 4 cm".split()), 1)  # đếm trên dãy từ đầy đủ, không bỏ số rồi mới đếm
        self.assertAlmostEqual(duplicate_sentence_ratio("Câu này đủ dài để được tính. Câu này đủ dài để được tính. Một câu khác hẳn, cũng đủ dài."), 1 / 3)
        self.assertEqual(duplicate_ngram_ratio("a b c a b c".split(), 3), 0.25)
        self.assertEqual(duplicate_ngram_ratio("một hai".split(), 5), 0.0)

    def test_each_turn_is_checked_separately(self):
        record = {r["id"]: r for r in fixture("repetition")}["rep-nhieu-luot"]
        _, answers = conversation(record)  # cả 4 lượt assistant nối lại thì lặp quá ngưỡng, nhưng từng lượt thì không
        words = answers.casefold().split()
        self.assertGreater(duplicate_ngram_ratio(words, 5), RepetitionFilter().max_duplicate_ngram_ratio)
        self.assertIsNone(RepetitionFilter().check(record))


class NearDuplicateTests(FixtureCase):
    def test_near_duplicate_fixture(self):
        records = fixture("near_duplicate")  # id xếp đúng thứ tự trong file: pipeline sắp theo id rồi giữ bản ghi đứng trước
        duplicates = self.config.near_duplicate.find(records)
        self.assertEqual({records[index]["id"]: "near_duplicate" for index in duplicates}, {key: value for key, value in expected(records).items() if value != "keep"})
        self.assertEqual({records[index]["id"]: records[original]["id"] for index, (original, _) in duplicates.items()},
                         {"dup-2-sua-mot-tu": "dup-1-goc", "dup-3-chi-khac-dau-cau": "dup-1-goc", "dup-7-chao-cham-than": "dup-6-chao"})
        self.assertTrue(all(similarity >= self.config.near_duplicate.threshold for _, similarity in duplicates.values()))

    def test_minhash_estimates_jaccard(self):
        records = fixture("near_duplicate"); minhash = MinHash(128, 17)
        sets = [shingles("\n".join(conversation(record)), 3) for record in records]
        for left in range(len(sets)):
            for right in range(left + 1, len(sets)):
                with self.subTest(f"{records[left]['id']} / {records[right]['id']}"):
                    self.assertLess(abs(estimated_jaccard(minhash.signature(sets[left]), minhash.signature(sets[right])) - jaccard(sets[left], sets[right])), 0.2)
        self.assertEqual(minhash.signature(sets[0]), MinHash(128, 17).signature(sets[0]))  # cùng seed thì cùng chữ ký
        self.assertNotEqual(minhash.signature(sets[0]), MinHash(128, 18).signature(sets[0]))

    def test_threshold_and_bands_come_from_config(self):
        records = fixture("near_duplicate")
        self.assertEqual(len(NearDuplicateFilter(threshold=0.3).find(records)), 4)  # hạ ngưỡng thì dòng giống một nửa cũng bị loại
        with self.assertRaisesRegex(ValueError, "chia hết"): NearDuplicateFilter(permutations=100, bands=32).find(records)


class ConfigTests(unittest.TestCase):
    def test_shipped_config_matches_code_defaults(self):
        self.assertEqual(QualityConfig.from_file(DEFAULT_QUALITY), QualityConfig())
        self.assertEqual(set(json.loads(DEFAULT_QUALITY.read_text(encoding="utf-8"))) - {"_comment"}, set(FILTERS))

    def test_unknown_or_invalid_settings_are_rejected(self):
        for value, message in (({"langauge": {}}, "nhóm lạ"), ({"length": {"max_chars": 10}}, "khóa lạ"), ({"near_duplicate": {"bands": 30}}, "chia hết"), ({"near_duplicate": {"threshold": 1.5}}, "threshold")):
            with self.subTest(message), self.assertRaisesRegex(ValueError, message): QualityConfig.from_dict(value)


class PipelineTests(unittest.TestCase):
    SOURCES = [FIXTURES / f"{name}.jsonl" for name in FILTERS]

    def test_manifest_has_statistics_before_and_after(self):
        records = [record for source in self.SOURCES for record in load_records(source)]
        wanted = expected(records)
        with tempfile.TemporaryDirectory() as directory:
            manifest = build_dataset(self.SOURCES, directory, "v", {"seed": 1, "formats": ["sft"]}, [], QualityConfig.from_file())
            written = json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))
            kept = {record["id"] for record in load_records(Path(directory) / "sft.jsonl")}
            rejected = {record["id"]: record["verification"] for record in load_records(Path(directory) / "rejected.jsonl")}
        quality = written["quality"]
        self.assertEqual(quality, json.loads(json.dumps(manifest["quality"])))
        self.assertTrue(quality["enabled"])
        self.assertEqual(quality["removed"], {name: list(wanted.values()).count(name) for name in FILTERS})
        self.assertEqual((quality["before"]["examples"], quality["after"]["examples"]), (len(records), list(wanted.values()).count("keep")))
        self.assertEqual(sum(quality["before"]["languages"].values()), len(records))
        self.assertIn("vi-khong-dau", quality["before"]["languages"]); self.assertNotIn("vi-khong-dau", quality["after"]["languages"])
        self.assertGreater(quality["before"]["characters"]["max"], quality["after"]["characters"]["max"])  # dòng quá dài đã bị loại
        self.assertTrue(0 < quality["after"]["vietnamese_share"] <= 1)
        self.assertEqual(quality["config"], json.loads(json.dumps(asdict(QualityConfig.from_file()))))
        self.assertEqual(kept, {key for key, value in wanted.items() if value == "keep"})
        self.assertEqual({key: value["quality_filter"] for key, value in rejected.items()}, {key: value for key, value in wanted.items() if value != "keep"})
        self.assertTrue(all(value["quality_reason"] for value in rejected.values()))
        self.assertEqual(written["statistics"]["examples"], len(kept))

    def test_filters_are_off_unless_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = build_dataset(self.SOURCES, directory, "v", {"seed": 1, "formats": ["sft"]}, [])
        self.assertEqual(manifest["quality"], {"enabled": False})
        self.assertEqual(manifest["statistics"]["examples"], sum(len(load_records(source)) for source in self.SOURCES))

    def test_each_filter_can_be_disabled(self):
        records = [record for source in self.SOURCES for record in load_records(source)]
        config = QualityConfig.from_file()
        for name in FILTERS:
            with self.subTest(name):
                off = replace(config, **{name: replace(getattr(config, name), enabled=False)})
                _, _, report = apply_quality_filters(records, off)
                self.assertEqual(report["removed"][name], 0)
                self.assertTrue(all(report["removed"][other] > 0 for other in FILTERS if other != name))

    def test_hf_mix_filters_by_default(self):
        presets = {load_preset(name)["hf_dataset"]["name"]: name for name in ("code", "reasoning", "vietnamese")}
        loader = lambda name, subset=None, **kwargs: iter(load_records(ROOT / "tests" / "fixtures" / "presets" / f"{presets[name]}.jsonl"))
        with tempfile.TemporaryDirectory() as directory:
            on = prepare_hf_mix({"code": 0.5, "reasoning": 0.25, "vietnamese": 0.25}, directory, loader, total=8)
            off = prepare_hf_mix({"code": 0.5, "reasoning": 0.25, "vietnamese": 0.25}, directory, loader, total=8, quality=False)
        self.assertTrue(on["quality"]["enabled"]); self.assertEqual(on["quality"]["before"]["examples"], 8)
        self.assertEqual(on["quality"]["before"]["languages"].get("vi"), 2)  # 2 dòng của preset vietnamese nhận đúng là tiếng Việt
        self.assertEqual(off["quality"], {"enabled": False})

    def test_command_line_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "build.json"; config.write_text('{"seed": 1, "formats": ["sft"]}', encoding="utf-8")
            result = subprocess.run([sys.executable, "-m", "local_ai.data", "build", *map(str, self.SOURCES), "--output", str(Path(directory) / "out"), "--version", "v", "--config", str(config), "--quality"],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["quality"]["removed"]["near_duplicate"], 3)
        for flag, shown in (([], "configs/datasets/quality.json"), (["--no-quality"], "tắt")):
            with self.subTest(shown):
                result = subprocess.run([sys.executable, "-m", "local_ai.data", "hf-sft", "--preset", "code", "--total", "10", "--dry-run", *flag], cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout)["quality"], shown)


if __name__ == "__main__":
    unittest.main()
