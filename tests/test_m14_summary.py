"""Mốc M14: tổng kết tuần 2. README khớp code: mọi lệnh, notebook, cấu hình train và tài liệu đều được nhắc tới;
nút Colab trỏ đúng file; bảng tiến độ tuần 2 đủ 7 mốc; lộ trình đề xuất 7 mốc cho tuần 3."""
import re
import unittest
from pathlib import Path

from local_ai.agents.tasks import load_tasks

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
# Từ lượt rà soát tuần 3 (26/9), kế hoạch tuần 1–2 được chép nguyên văn từ TASKS.md sang archive/tasks-tuan-1-2.md (chủ repo cho phép đổi chỗ đọc).
TASKS = (ROOT / "archive" / "tasks-tuan-1-2.md").read_text(encoding="utf-8")
COLAB = "https://colab.research.google.com/github/hytmk2912/Huyen/blob/main/"


def command_modules() -> list[str]:
    """Mọi module chạy được bằng `python -m`: file có `if __name__ == "__main__"`, hoặc `__main__.py` của một package."""
    modules = []
    for path in sorted((ROOT / "local_ai").rglob("*.py")):
        if '__name__ == "__main__"' not in path.read_text(encoding="utf-8"): continue
        parts = path.relative_to(ROOT).with_suffix("").parts
        modules.append(".".join(parts[:-1] if parts[-1] == "__main__" else parts))
    return modules


class ReadmeMatchesCodeTests(unittest.TestCase):
    def test_every_command_is_documented(self):
        modules = command_modules()
        self.assertGreaterEqual(len(modules), 10)
        for module in modules:
            with self.subTest(module): self.assertRegex(README, re.escape(module) + r"(?![\w.])")

    def test_every_notebook_has_a_colab_button(self):
        notebooks = sorted(path.relative_to(ROOT).as_posix() for path in (ROOT / "notebooks").glob("*.ipynb"))
        self.assertEqual(notebooks, ["notebooks/agent_colab.ipynb", "notebooks/train_colab.ipynb"])
        for notebook in notebooks:
            with self.subTest(notebook): self.assertIn(f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB}{notebook})", README)
        for link in re.findall(re.escape(COLAB) + r"([^)\s]+)", README):  # mọi nút Colab trỏ tới file có thật
            with self.subTest(link): self.assertTrue((ROOT / link).is_file())

    def test_configs_docs_and_data_are_mentioned(self):
        names = [path.relative_to(ROOT).as_posix() for path in sorted((ROOT / "configs" / "training").glob("*.json"))]
        names += ["configs/datasets/quality.json", "configs/tools/terminal.json", "configs/runtime/gateway.json", "data/eval/agent_tasks_v1.jsonl"]
        names += [path.relative_to(ROOT).as_posix() for path in sorted((ROOT / "docs").glob("*.md"))]
        for name in names:
            with self.subTest(name): self.assertIn(name, README)

    def test_status_table_covers_both_weeks(self):
        status = README.split("## Trạng thái hiện tại", 1)[1].split("\n## ", 1)[0]  # rà soát tuần 3: trước là "## Trạng thái sau tuần 1 và tuần 2"
        for part in ("Dữ liệu", "Fine-tune", "Eval", "Model qua server local", "Agent", "Runtime gộp từ repo Agent (tuần 2)", "Notebook Colab (tuần 2)"):
            with self.subTest(part): self.assertIn(f"| {part} |", status)
        self.assertIn("Chưa chạy trên Colab thật", status)
        self.assertIn(f"{len(load_tasks())} nhiệm vụ mẫu", status)

    def test_roadmap_proposes_seven_milestones_for_week_three(self):
        roadmap = README.split("## Lộ trình tiếp theo (đề xuất, chưa làm)", 1)[1]
        self.assertEqual(re.findall(r"(?m)^\d+\. \*\*(M\d+)", roadmap), [f"M{number}" for number in range(15, 22)])


class TasksTests(unittest.TestCase):
    def test_week_two_milestones_are_all_done(self):
        week_two = TASKS.split("# Nhiệm vụ tuần 2", 1)[1].split("# Nhiệm vụ tuần 1", 1)[0]
        rows = re.findall(r"(?m)^\| (M\d+) \| [^|]+ \| ([^|]+) \| ([^|]+) \|", week_two)
        self.assertEqual([row[0] for row in rows], [f"M{number}" for number in range(8, 15)])
        for milestone, status, progress in rows:
            with self.subTest(milestone): self.assertEqual((status.strip(), progress.strip().endswith("(100%)")), ("**Xong**", True))


if __name__ == "__main__":
    unittest.main()
