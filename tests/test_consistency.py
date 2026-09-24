"""Kiểm tra repo thống nhất: cấu hình dataset, thư mục dữ liệu SFT, lệnh và chữ tiếng Việt cho người dùng."""
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from local_ai.data.hub import DEFAULT_SFT_DIR

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
VIETNAMESE = re.compile(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", re.IGNORECASE)
MODEL_FACING = {"Only numeric arithmetic expressions are allowed"}  # kết quả công cụ gửi cho model, được phép tiếng Anh


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class DatasetConfigTests(unittest.TestCase):
    def test_template_and_presets_follow_the_same_rules(self):
        paths = [ROOT / "configs" / "datasets" / "hf_sft.json", *sorted((ROOT / "configs" / "datasets" / "presets").glob("*.json"))]
        for path in paths:
            with self.subTest(path.name):
                config = read_json(path); spec = config["hf_dataset"]
                self.assertIs(spec["streaming"], True)  # CLAUDE.md: dataset thật chỉ đọc streaming với limit nhỏ
                self.assertTrue(0 < spec["limit"] <= 1000)
                self.assertEqual(spec["license"]["status"], "cần kiểm tra lại trên dataset card")
                self.assertIn("language", spec); self.assertIn("description", config)
                self.assertEqual(config["eval_sources"], ["data/eval/seed_eval.jsonl", "data/eval/eval_v1.jsonl"])

    def test_sft_directory_is_the_same_everywhere(self):
        self.assertEqual(read_json(ROOT / "configs" / "datasets" / "hf_sft.json")["output_dir"], DEFAULT_SFT_DIR)
        for path in (ROOT / "configs" / "training").glob("*.json"):
            with self.subTest(path.name):
                self.assertEqual(read_json(path)["dataset_path"], f"{DEFAULT_SFT_DIR}/sft.jsonl")
        self.assertNotIn("hf_mix", README)
        for command in re.findall(r"python -m local_ai\.data hf-sft --preset[^\n`]*", README):
            with self.subTest(command):
                self.assertIn(f"--output {DEFAULT_SFT_DIR}", command)


class CommandTests(unittest.TestCase):
    def test_every_data_subcommand_has_help(self):
        text = subprocess.run([sys.executable, "-m", "local_ai.data", "--help"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        names = re.search(r"\{([a-z,-]+)\}", text).group(1).split(",")
        self.assertEqual(len(names), 9)
        for name in names:
            with self.subTest(name):
                self.assertRegex(text, rf"(?m)^\s+{re.escape(name)}\s+\S")

    def test_default_paths_work_outside_the_repo_folder(self):
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        with tempfile.TemporaryDirectory() as directory:
            vram = subprocess.run([sys.executable, "-m", "local_ai.models.vram"], cwd=directory, env=env, capture_output=True, text=True)
            self.assertEqual(vram.returncode, 0, vram.stderr); self.assertIn("| primary |", vram.stdout)
            evaluation = subprocess.run([sys.executable, "-m", "local_ai.evaluation", "--scripted", "--output", str(Path(directory) / "eval")], cwd=directory, env=env, capture_output=True, text=True)
            self.assertEqual(evaluation.returncode, 0, evaluation.stderr); self.assertIn("30/30", evaluation.stdout)


class LanguageTests(unittest.TestCase):
    def test_user_facing_text_is_vietnamese(self):
        for path in sorted((ROOT / "local_ai").rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            texts = re.findall(r'(?:help|description)="([^"]+)"', source) + re.findall(r'raise \w+\(f?"([^"]+)"', source)
            for text in texts:
                if text in MODEL_FACING or not re.search(r"[A-Za-z]", re.sub(r"\{[^}]*\}", "", text)): continue  # chuỗi chỉ chuyển tiếp lỗi gốc
                with self.subTest(f"{path.relative_to(ROOT)}: {text[:60]}"):
                    self.assertRegex(text, VIETNAMESE)


if __name__ == "__main__":
    unittest.main()
