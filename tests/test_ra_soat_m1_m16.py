"""Lượt rà soát M1–M16 (25/9): các sửa nhỏ làm được ngay, không thuộc mốc nào.

- secret-scan bắt thêm kiểu mật khẩu từng lọt vào lịch sử commit (biến shell viết HOA có chữ PASS, kể cả giá trị mặc định
  `${BIẾN:-...}`). Chuỗi giống khóa trong test được ghép lúc chạy, để chính file test không bị secret-scan báo.
- Nạp model dùng `dtype` thay cho `torch_dtype` (transformers 5.x báo `torch_dtype` đã cũ).
- `.gitignore` bỏ qua file trọng số `*.bin` và `.ruff_cache/`.
- Giấy phép 3 dataset preset được ghi trong `docs/GIAY_PHEP_DATASET.md`.
"""
import json
import tempfile
import unittest
from pathlib import Path

from local_ai.data.secrets import SECRET, scan_secrets

ROOT = Path(__file__).resolve().parent.parent
NAME = "SYNC" + "_PASS"
VALUE = "x7Kq" + "Lm2Pw9"  # giá trị giả, chỉ để thử mẫu tìm kiếm


class SecretPatternTests(unittest.TestCase):
    def test_uppercase_shell_password_is_found(self):
        for line in (f'{NAME}="${{{NAME}:-{VALUE}}}"', f"{NAME}={VALUE}", f"export RSYNC{NAME[4:]}WORD='{VALUE}'"):
            with self.subTest(line): self.assertIsNotNone(SECRET.search(line))

    def test_value_read_from_other_variable_is_not_a_secret(self):
        for line in (f'{NAME}="${{OTHER_VAR}}"', "SSHPASS=$REMOTE_PASS", f"{NAME}=$1", "bypass=abcdefghijk", f"{NAME.lower()}=abcdefghijk"):
            with self.subTest(line): self.assertIsNone(SECRET.search(line))

    def test_scan_reports_shell_script_with_literal_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sync.sh").write_text(f'#!/bin/sh\n{NAME}="${{{NAME}:-{VALUE}}}"\nrsync --password-file=- "$SRC" huyen@host::data\n', encoding="utf-8")
            (root / "safe.sh").write_text(f'#!/bin/sh\n{NAME}="${{{NAME}:?chưa đặt biến}}"\n', encoding="utf-8")
            self.assertEqual(scan_secrets(root), [str(root / "sync.sh")])


class ReviewFixTests(unittest.TestCase):
    def test_models_load_with_dtype_not_torch_dtype(self):
        for path in (ROOT / "local_ai").rglob("*.py"):
            with self.subTest(str(path.relative_to(ROOT))): self.assertNotIn('"torch_dtype"', path.read_text(encoding="utf-8"))

    def test_gitignore_skips_weights_and_tool_cache(self):
        lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        for pattern in ("*.bin", "*.safetensors", "*.gguf", ".ruff_cache/", ".env"):
            with self.subTest(pattern): self.assertIn(pattern, lines)

    def test_dataset_licenses_are_documented(self):
        doc = (ROOT / "docs" / "GIAY_PHEP_DATASET.md").read_text(encoding="utf-8")
        for preset in ("code", "reasoning", "vietnamese"):
            spec = json.loads((ROOT / "configs" / "datasets" / "presets" / f"{preset}.json").read_text(encoding="utf-8"))["hf_dataset"]
            with self.subTest(preset):
                self.assertIn(f"`{spec['name']}`", doc)
                self.assertIn(spec["license"]["name"], doc)
                self.assertEqual(spec["license"]["status"], "cần kiểm tra lại trên dataset card")  # chủ repo chưa xác nhận
        self.assertIn("CC BY-NC 4.0", doc)
        comment = json.loads((ROOT / "configs" / "datasets" / "presets" / "vietnamese.json").read_text(encoding="utf-8"))["_comment"]
        self.assertIn("không dùng thương mại", comment); self.assertIn("docs/GIAY_PHEP_DATASET.md", comment)
        self.assertIn("(docs/GIAY_PHEP_DATASET.md)", (ROOT / "README.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
