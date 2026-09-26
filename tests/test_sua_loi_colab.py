"""Sửa lỗi notebook train_colab mà chủ repo gặp khi chạy thật trên Colab T4 (25–26/9).

- Colab cài sẵn torchao, bản này làm lỗi khi train với các thư viện đã ghim; chủ repo phải gỡ tay khi chạy smoke và light.
  Bước 3 của notebook nay tự gỡ torchao sau khi cài thư viện.
"""
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def cell_source(notebook: str, cell_id: str) -> str:
    data = json.loads((ROOT / "notebooks" / notebook).read_text(encoding="utf-8"))
    return "".join(next(cell for cell in data["cells"] if cell.get("id") == cell_id)["source"])


class TorchaoTests(unittest.TestCase):
    def test_step_three_removes_torchao_after_installing_pinned_libraries(self):
        source = cell_source("train_colab.ipynb", "buoc-3-cai-dat")
        steps = re.findall(r'run\("([^"]+)", "([^"]+)"\)', source)
        commands = [command for command, _ in steps]
        install = next(index for index, command in enumerate(commands) if command.startswith("pip install"))
        uninstall = commands.index("pip uninstall -y -q torchao")
        self.assertGreater(uninstall, install)  # cài xong mới gỡ, để pip không cài lại torchao
        self.assertEqual(steps[uninstall][1], "Bước 3 (gỡ torchao)")
        self.assertIn("torchao", source.split("run(\"pip uninstall")[0])  # có chú thích tiếng Việt giải thích lý do
        self.assertNotIn("torchao", " ".join(command for command in commands if command.startswith("pip install")))


if __name__ == "__main__":
    unittest.main()
