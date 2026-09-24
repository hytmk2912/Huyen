"""Mốc M4: preset dataset Hugging Face (code, reasoning, tiếng Việt), trộn nhiều preset theo tỉ lệ.

Test dùng loader giả đọc fixture trong tests/fixtures/presets/, không cần mạng hay thư viện datasets.
"""
import contextlib
import io
import itertools
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_ai.data import hub
from local_ai.data.core import load_records, prepare_format, validate_records
from local_ai.data.hub import list_presets, load_hf_rows, load_preset, map_rows, mix_counts, parse_mix, prepare_hf_mix, prepare_hf_sft

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).parent / "fixtures" / "presets"
PRESETS = ("code", "reasoning", "vietnamese")


def fixture_loader(calls):
    """Loader giả cùng tham số với datasets.load_dataset; trả các dòng fixture của preset tương ứng."""
    names = {load_preset(name)["hf_dataset"]["name"]: name for name in PRESETS}
    def load(name, subset=None, **kwargs):
        calls.append({"name": name, "subset": subset, **kwargs})
        return iter(load_records(FIXTURES / f"{names[name]}.jsonl"))
    return load


def run_preset(name, directory, calls):
    manifest = prepare_hf_sft(load_preset(name), directory, fixture_loader(calls))
    return manifest, load_records(Path(directory) / "sft.jsonl"), load_records(Path(directory) / "rejected.jsonl")


class PresetFileTests(unittest.TestCase):
    def test_three_presets_point_to_real_datasets(self):
        presets = {item["name"]: item for item in list_presets()}
        self.assertEqual(set(presets), set(PRESETS))
        self.assertEqual((presets["code"]["domain"], presets["reasoning"]["domain"], presets["vietnamese"]["language"]), ("coding", "math_logic", "vi"))
        for name, item in presets.items():
            with self.subTest(name):
                self.assertRegex(item["dataset"], r"^[\w.-]+/[\w.-]+$")
                self.assertRegex(item["revision"], r"^[0-9a-f]{40}$")  # commit cố định, không dùng "main"
                self.assertTrue(item["license"] and item["description"])
                self.assertEqual(item["license_status"], "cần kiểm tra lại trên dataset card")

    def test_presets_stream_with_small_limit(self):
        for name in PRESETS:
            with self.subTest(name):
                spec = load_preset(name)["hf_dataset"]
                self.assertIs(spec["streaming"], True)
                self.assertTrue(0 < spec["limit"] <= 1000)

    def test_loader_reads_only_limit_rows(self):
        consumed = []
        def endless(name, subset=None, **kwargs):
            self.assertIs(kwargs["streaming"], True)
            for index in itertools.count():
                consumed.append(index); yield {"id": index, "instruction": f"q{index}", "response": f"a{index}"}
        rows = load_hf_rows({**load_preset("code")["hf_dataset"], "limit": 7}, endless)
        self.assertEqual((len(rows), len(consumed)), (7, 7))


class PresetMappingTests(unittest.TestCase):
    def test_each_preset_maps_columns_and_calls_loader_with_streaming(self):
        expected = {"code": (4, 1), "reasoning": (3, 1), "vietnamese": (3, 1)}
        for name in PRESETS:
            with self.subTest(name), tempfile.TemporaryDirectory() as directory:
                calls = []
                manifest, sft, rejected = run_preset(name, directory, calls)
                spec = load_preset(name)["hf_dataset"]
                self.assertEqual({key: calls[0][key] for key in ("name", "subset", "revision", "streaming")}, {key: spec[key] for key in ("name", "subset", "revision", "streaming")})
                self.assertEqual((len(sft), len(rejected)), expected[name])
                self.assertEqual(manifest["statistics"]["domains"], {spec["domain"]: expected[name][0]})

    def test_code_rows(self):
        with tempfile.TemporaryDirectory() as directory: _, sft, _ = run_preset("code", directory, [])
        first = next(row for row in sft if row["id"].endswith("-11"))
        self.assertEqual(first["messages"], [{"role": "user", "content": "Write a function `add(a, b)` that returns the sum."}, {"role": "assistant", "content": "def add(a, b):\n    return a + b"}])

    def test_reasoning_is_kept_in_think_block(self):
        with tempfile.TemporaryDirectory() as directory: _, sft, rejected = run_preset("reasoning", directory, [])
        first = next(row for row in sft if row["id"].endswith("-u1"))
        self.assertEqual(first["messages"][1]["content"], "<think>\nMultiply first: 3 * 4 = 12, then 2 + 12 = 14.\n</think>\n\n14")
        self.assertIn("missing:expected_output", rejected[0]["verification"]["schema_errors"])  # bài không có đáp án bị loại

    def test_multi_turn_conversation_is_kept_with_system(self):
        with tempfile.TemporaryDirectory() as directory: _, sft, rejected = run_preset("vietnamese", directory, [])
        first = next(row for row in sft if row["id"].endswith("-v1"))
        self.assertEqual(first["messages"], [
            {"role": "system", "content": "Bạn là trợ lý tiếng Việt."}, {"role": "user", "content": "Thủ đô của Pháp là gì?"}, {"role": "assistant", "content": "Paris."},
            {"role": "user", "content": "Còn của Đức?"}, {"role": "assistant", "content": "Berlin."}])
        self.assertIn("invalid:messages", rejected[0]["verification"]["schema_errors"])  # kết thúc bằng câu hỏi thì không train được

    def test_missing_or_nan_id_falls_back_to_row_number(self):
        spec = {"name": "org/ds", "license": {"name": "CC0-1.0"}, "mapping": {"id": "uuid", "input": "q", "expected_output": "a"}}
        rows = [{"uuid": float("nan"), "q": "1", "a": "1"}, {"uuid": None, "q": "2", "a": "2"}, {"uuid": "NaN", "q": "3", "a": "3"}, {"uuid": "u4", "q": "4", "a": "4"}]
        self.assertEqual([record["id"] for record in map_rows(rows, spec, "v")], ["org-ds-0", "org-ds-1", "org-ds-2", "org-ds-u4"])

    def test_context_is_put_before_question(self):
        spec = {"name": "org/ctx", "license": {"name": "CC0-1.0"}, "mapping": {"input": "q", "expected_output": "a", "context": "c"}}
        valid, _ = validate_records(map_rows([{"q": "Câu hỏi?", "a": "Đáp án.", "c": "Đoạn văn."}], spec, "v"))
        self.assertEqual(prepare_format(valid, "sft")[0]["messages"][0]["content"], "Đoạn văn.\n\nCâu hỏi?")


class MixTests(unittest.TestCase):
    def test_parse_mix_normalizes_weights(self):
        self.assertEqual(parse_mix(["code:2", "vietnamese:2"]), {"code": 0.5, "vietnamese": 0.5})
        self.assertEqual(parse_mix(["code"]), {"code": 1.0})
        for values in (["code:0"], ["code", "code"], ["code:x"], []):
            with self.subTest(values), self.assertRaises(ValueError): parse_mix(values)

    def test_counts_respect_ratio_and_limits(self):
        self.assertEqual(mix_counts({"code": 0.4, "reasoning": 0.3, "vietnamese": 0.3}, {"code": 1000, "reasoning": 1000, "vietnamese": 1000}), {"code": 1000, "reasoning": 750, "vietnamese": 750})
        self.assertEqual(mix_counts({"code": 0.5, "vietnamese": 0.5}, {"code": 1000, "vietnamese": 1000}, total=10), {"code": 5, "vietnamese": 5})
        with self.assertRaisesRegex(ValueError, "tối đa"): mix_counts({"code": 1.0}, {"code": 10}, total=20)

    def test_mix_writes_one_sft_file_with_ratio(self):
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            manifest = prepare_hf_mix({"code": 0.5, "reasoning": 0.25, "vietnamese": 0.25}, directory, fixture_loader(calls), total=8)
            sft = load_records(Path(directory) / "sft.jsonl")
            self.assertEqual(sorted(path.name for path in Path(directory).glob("*.jsonl")), ["raw.jsonl", "rejected.jsonl", "sft.jsonl", "train.jsonl"])
        self.assertEqual([call["name"] for call in calls], [load_preset(name)["hf_dataset"]["name"] for name in PRESETS])
        self.assertEqual({name: item["rows"] for name, item in manifest["configuration"]["mix"].items()}, {"code": 4, "reasoning": 2, "vietnamese": 2})
        self.assertEqual(sorted(manifest["statistics"]["sources"].values()), [2, 2, 4])
        self.assertEqual(len(sft), 8)


class CommandTests(unittest.TestCase):
    def run_main(self, argv):
        from local_ai.data.__main__ import main
        stdout = io.StringIO()
        with mock.patch.object(sys, "argv", ["python -m local_ai.data", *argv]), mock.patch.object(hub, "load_hf_rows", lambda spec, loader=None: load_hf_rows(spec, fixture_loader([]))), contextlib.redirect_stdout(stdout):
            main()
        return stdout.getvalue()

    def test_list_presets_command(self):
        result = subprocess.run([sys.executable, "-m", "local_ai.data", "list-presets"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in PRESETS: self.assertRegex(result.stdout, rf"(?m)^{name}: ")
        self.assertEqual(result.stdout.count("cần kiểm tra lại trên dataset card"), len(PRESETS))

    def test_hf_sft_accepts_config_or_several_presets(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_main(["hf-sft", "--config", "configs/datasets/presets/vietnamese.json", "--output", f"{directory}/one"])
            self.assertEqual(len(load_records(Path(directory) / "one" / "sft.jsonl")), 3)
            output = self.run_main(["hf-sft", "--preset", "code:0.5", "--preset", "vietnamese:0.5", "--total", "4", "--output", f"{directory}/mix"])
            self.assertEqual(len(load_records(Path(directory) / "mix" / "sft.jsonl")), 4)
        self.assertEqual(json.loads(output)["configuration"]["mix"]["code"]["rows"], 2)
        self.assertFalse(hub.real_stream_used())


class ReadmeTests(unittest.TestCase):
    def test_readme_explains_presets_and_license_check(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for phrase in ("list-presets", "--preset code:", "configs/datasets/presets/", "giấy phép", "dataset card"):
            self.assertIn(phrase, readme)
        self.assertTrue(re.search(r"hf-sft --config configs/datasets/presets/\w+\.json", readme))


if __name__ == "__main__":
    unittest.main()
