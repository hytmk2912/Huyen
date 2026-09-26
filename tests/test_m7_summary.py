"""Mốc M7: README khớp code (lệnh, cờ, số VRAM, đường dẫn adapter), lộ trình đề xuất; secret-scan chỉ quét file có thể bị commit."""
import json
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from functools import lru_cache
from pathlib import Path

from local_ai.config.settings import find_model_config
from local_ai.data.secrets import scan_secrets
from local_ai.models.vram import training_gb, weights_gb

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
PLATFORM = ROOT / "configs" / "models" / "platform.json"


def readme_commands() -> list[list[str]]:
    """Mọi lệnh `python -m local_ai...` trong README: trong khối ```bash và trong code inline (ví dụ bảng train trên GPU).
    Bỏ chú thích và dấu [ ] của tham số tùy chọn."""
    lines = [line.split("  #")[0].strip() for block in re.findall(r"```bash\n(.*?)```", README, re.DOTALL) for line in block.splitlines()]
    lines += re.findall(r"(?<!`)`(python -m local_ai[^`]*)`", README)
    return [shlex.split(line.replace("[", " ").replace("]", " ")) for line in lines if line.startswith("python -m local_ai")]


@lru_cache(maxsize=None)
def help_text(module: str, subcommand: str | None) -> str:
    argv = [sys.executable, "-m", module, *([subcommand] if subcommand else []), "--help"]
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout


def vi(value: float) -> str:
    return f"{value:.1f}".replace(".", ",")


class ReadmeMatchesCodeTests(unittest.TestCase):
    def test_every_readme_command_uses_existing_flags(self):
        commands = readme_commands()
        self.assertGreaterEqual(len(commands), 15)
        for tokens in commands:
            with self.subTest(" ".join(tokens)):
                module = tokens[2]
                subcommand = tokens[3] if module == "local_ai.data" and len(tokens) > 3 else None
                text = help_text(module, subcommand)
                for flag in (token.split("=")[0] for token in tokens if token.startswith("--")):
                    self.assertRegex(text, re.escape(flag) + r"(?![\w-])")

    def test_vram_numbers_in_readme_match_estimator(self):
        for name, precision in (("smoke", "bf16"), ("light", "bf16"), ("light", "4bit"), ("primary", "4bit"), ("primary", "bf16")):
            with self.subTest(f"{name} {precision}"):
                self.assertIn(f"{vi(training_gb(find_model_config(PLATFORM, name).params_b, precision))} GB", README)
        primary = find_model_config(PLATFORM, "primary").params_b
        for precision in ("bf16", "4bit"):
            self.assertIn(vi(weights_gb(primary, precision)), README)

    def test_post_training_models_point_to_training_outputs(self):
        def output_dir(name): return json.loads((ROOT / "configs" / "training" / name).read_text(encoding="utf-8"))["output_dir"]
        self.assertEqual(find_model_config(PLATFORM, "smoke-lora").adapter_path, f"{output_dir('sft.json')}/adapter")
        self.assertEqual(find_model_config(PLATFORM, "primary-qlora").adapter_path, f"{output_dir('qlora_primary.json')}/adapter")
        light = find_model_config(PLATFORM, "light-lora").adapter_path
        self.assertIn(f"--output-dir {light.rsplit('/adapter', 1)[0]}", README)
        self.assertEqual(find_model_config(PLATFORM, "primary-qlora").quantization, "4bit")
        dataset_paths = {json.loads(path.read_text(encoding="utf-8"))["dataset_path"] for path in (ROOT / "configs" / "training").glob("*.json")}
        self.assertEqual(dataset_paths, {"data/processed/hf_sft/sft.jsonl"})
        self.assertIn("--output data/processed/hf_sft", README)  # lệnh tạo dữ liệu ghi đúng chỗ các cấu hình train đọc

    def test_readme_has_status_gpu_plan_and_proposed_roadmap(self):
        for phrase in ("## Trạng thái hiện tại", "## Train trên GPU: smoke → light → primary",  # rà soát tuần 3: tiêu đề cũ "## Trạng thái sau tuần 1" đổi thành "## Trạng thái hiện tại"
                        "## Lộ trình tiếp theo (đề xuất, chưa làm)", "chưa chạy thử", "TASKS.md"):
            self.assertIn(phrase, README)


class SecretScanTests(unittest.TestCase):
    def test_gitignored_files_are_not_scanned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            (root / ".gitignore").write_text("data/processed/\n.env\n", encoding="utf-8")
            (root / "data" / "processed").mkdir(parents=True)
            (root / "data" / "processed" / "train.jsonl").write_text('{"code": "pass' + 'word = get_password()"}\n', encoding="utf-8")  # ghép chuỗi để secret-scan của repo không bắt chính file test
            (root / ".env").write_text("HF_TOKEN=hf_" + "b" * 30 + "\n", encoding="utf-8")
            (root / "config.py").write_text("api" + "_key = 'abcdefghijkl'\n", encoding="utf-8")
            self.assertEqual(scan_secrets(root), [str(root / "config.py")])


if __name__ == "__main__":
    unittest.main()
