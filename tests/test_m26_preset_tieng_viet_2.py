"""Mốc M26: preset tiếng Việt thứ 2 (`vietnamese_aya`), lấy câu hỏi và câu trả lời tiếng Việt do người viết trong
`CohereLabs/aya_dataset` (Apache-2.0).

Test dùng loader giả đọc fixture `tests/fixtures/presets/vietnamese_aya.jsonl` (dòng thứ 4 là một dòng thật của Aya,
Apache-2.0; các dòng khác tự viết), không cần mạng hay thư viện datasets.
"""
import contextlib
import io
import itertools
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from local_ai.data.core import load_records
from local_ai.data.hub import list_presets, load_hf_rows, load_preset, prepare_hf_mix, prepare_hf_sft, validate_spec

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "presets"
PRESET = "vietnamese_aya"
LICENSE_DOC = (ROOT / "docs" / "GIAY_PHEP_DATASET.md").read_text(encoding="utf-8")


def fixture_loader(calls):
    """Loader giả cùng tham số với datasets.load_dataset; trả các dòng fixture của preset có cùng tên dataset."""
    names = {load_preset(name)["hf_dataset"]["name"]: name for name in ("code", "reasoning", "vietnamese", PRESET)}
    def load(name, subset=None, **kwargs):
        calls.append({"name": name, "subset": subset, **kwargs})
        return iter(load_records(FIXTURES / f"{names[name]}.jsonl"))
    return load


class PresetFileTests(unittest.TestCase):
    def test_preset_points_to_a_pinned_dataset_with_a_license_that_allows_use(self):
        config = load_preset(PRESET); spec = config["hf_dataset"]
        self.assertEqual((spec["name"], spec["revision"]), ("CohereLabs/aya_dataset", "f9ea04583f02a8f86404ff6c58bf75fe637df8a2"))
        self.assertEqual((spec["license"]["name"], spec["license"]["status"]), ("Apache-2.0", "cần kiểm tra lại trên dataset card"))
        self.assertEqual((spec["language"], spec["domain"], spec["task"]), ("vi", "chat", "sft"))
        self.assertIs(spec["streaming"], True)
        self.assertTrue(0 < spec["limit"] <= 1000)
        self.assertEqual(spec["where"], [["language_code", "vie"], ["annotation_type", "original-annotations"]])  # chỉ dòng tiếng Việt do người viết mới
        self.assertTrue(spec["limit"] < spec["scan_limit"] <= 50000)  # không đọc cả 202.362 dòng của dataset
        validate_spec(spec)
        self.assertIn("docs/GIAY_PHEP_DATASET.md", config["_comment"]); self.assertIn("kể cả thương mại", config["_comment"])
        listed = {item["name"]: item for item in list_presets()}[PRESET]
        self.assertEqual((listed["dataset"], listed["license"], listed["language"]), ("CohereLabs/aya_dataset", "Apache-2.0", "vi"))

    def test_license_is_written_in_the_license_doc(self):
        spec = load_preset(PRESET)["hf_dataset"]
        for phrase in (f"`{spec['name']}`", spec["license"]["name"], spec["revision"], "`original-annotations`", "`re-annotations`",
                       "whether academic or commercial", "Cá nhân và thương mại; nên ghi nguồn", "4.853"):
            with self.subTest(phrase): self.assertIn(phrase, LICENSE_DOC)
        row = next(line for line in LICENSE_DOC.splitlines() if line.startswith(f"| `{PRESET}` |"))
        self.assertIn("Apache-2.0", row)
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn(f"| `{PRESET}` (M26) | `{spec['name']}` |", readme)
        self.assertIn(f"--preset {PRESET}:0.3", readme)


class PresetMappingTests(unittest.TestCase):
    def test_only_vietnamese_rows_written_by_people_are_kept(self):
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            manifest = prepare_hf_sft(load_preset(PRESET), directory, fixture_loader(calls))
            sft, rejected = load_records(Path(directory) / "sft.jsonl"), load_records(Path(directory) / "rejected.jsonl")
        spec = load_preset(PRESET)["hf_dataset"]
        self.assertEqual({key: calls[0][key] for key in ("name", "subset", "revision", "streaming", "split")}, {key: spec[key] for key in ("name", "subset", "revision", "streaming", "split")})
        self.assertEqual((len(sft), len(rejected)), (3, 1))  # bỏ dòng tiếng Anh và dòng re-annotations; dòng thiếu câu trả lời bị loại
        self.assertIn("missing:expected_output", rejected[0]["verification"]["schema_errors"])
        self.assertEqual(manifest["statistics"]["domains"], {"chat": 3})
        questions = [row["messages"][0]["content"] for row in sft]
        self.assertEqual(questions, ["Con vật nào ứng với năm Tý trong 12 con giáp?", "Trong 5 phút, 5 máy tạo ra được 5 chi tiết. Hỏi trong bao nhiêu phút, 100 máy tạo ra được 100 chi tiết?",
                                     "Bánh chưng thường được gói vào dịp nào?"])
        self.assertEqual(sft[0]["messages"], [{"role": "user", "content": "Con vật nào ứng với năm Tý trong 12 con giáp?"}, {"role": "assistant", "content": "Năm Tý ứng với con chuột."}])
        self.assertEqual(manifest["quality"]["before"]["languages"], {"vi": 3})

    def test_mix_can_replace_the_first_vietnamese_preset(self):
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            manifest = prepare_hf_mix({"code": 0.5, PRESET: 0.5}, directory, fixture_loader(calls), total=6)
        self.assertEqual({name: item["rows"] for name, item in manifest["configuration"]["mix"].items()}, {"code": 3, PRESET: 3})
        self.assertEqual(manifest["configuration"]["mix"][PRESET]["dataset"], "CohereLabs/aya_dataset")
        self.assertEqual(manifest["statistics"]["sources"]["CohereLabs/aya_dataset"], 2)  # 3 dòng đọc vào, 1 dòng thiếu câu trả lời bị loại

    def test_cli_dry_run_accepts_the_new_preset(self):
        command = [sys.executable, "-m", "local_ai.data", "hf-sft", "--preset", "code:0.4", "--preset", "reasoning:0.3", "--preset", f"{PRESET}:0.3", "--total", "2000", "--dry-run"]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual((plan["presets"][PRESET]["dataset"], plan["presets"][PRESET]["rows"]), ("CohereLabs/aya_dataset", 600))


class RowFilterTests(unittest.TestCase):
    def test_where_keeps_matching_rows_and_scan_limit_bounds_reading(self):
        consumed = []
        def endless(name, subset=None, **kwargs):
            self.assertIs(kwargs["streaming"], True)
            for index in itertools.count():
                consumed.append(index)
                yield {"inputs": f"q{index}", "targets": f"a{index}", "language_code": "vie" if index % 10 == 0 else "eng", "annotation_type": "original-annotations"}
        spec = {**load_preset(PRESET)["hf_dataset"], "limit": 100, "scan_limit": 50}
        rows = load_hf_rows(spec, endless)
        self.assertEqual([row["inputs"] for row in rows], ["q0", "q10", "q20", "q30", "q40"])
        self.assertEqual(len(consumed), 50)  # đọc đúng scan_limit dòng rồi dừng, dù chưa đủ limit
        consumed.clear()
        self.assertEqual(len(load_hf_rows({**spec, "limit": 2}, endless)), 2)
        self.assertEqual(len(consumed), 11)  # đủ limit thì dừng sớm, không đọc hết scan_limit

    def test_presets_without_where_read_as_before(self):
        rows = load_hf_rows({"name": "org/ds", "limit": 3}, lambda *args, **kwargs: iter({"i": index} for index in range(10)))
        self.assertEqual(rows, [{"i": 0}, {"i": 1}, {"i": 2}])

    def test_bad_filter_settings_are_explained_in_vietnamese(self):
        base = {"name": "org/ds", "license": {"name": "Apache-2.0"}, "mapping": {"input": "q", "expected_output": "a"}}
        for extra, message in (({"where": [["lang", "vie"]]}, "scan_limit"), ({"where": [["lang", "vie"]], "scan_limit": 0}, "scan_limit"),
                               ({"where": [], "scan_limit": 10}, "where"), ({"where": {"lang": "vie"}, "scan_limit": 10}, "where"),
                               ({"where": [["lang"]], "scan_limit": 10}, "where"), ({"where": [["lang", ["vie"]]], "scan_limit": 10}, "where")):
            with self.subTest(extra), self.assertRaisesRegex(ValueError, message):
                validate_spec({**base, **extra})
        validate_spec({**base, "where": [["lang", "vie"]], "scan_limit": 10})


if __name__ == "__main__":
    unittest.main()
