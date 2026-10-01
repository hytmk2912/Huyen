"""Mốc M27: tổng kết tuần 4. README, `TASKS.md`, `memory.md` và `docs/ARCHITECTURE.md` khớp thực tế; có đề xuất tuần 5."""
import ast
import re
import unittest
from pathlib import Path

from local_ai.config.settings import find_model_config
from local_ai.data.hub import list_presets
from local_ai.models.vram import max_seq_len, training_gb
from local_ai.training.finetune import load_finetune_config

ROOT = Path(__file__).resolve().parent.parent
TASKS = (ROOT / "TASKS.md").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")
MEMORY = (ROOT / "memory.md").read_text(encoding="utf-8")
ARCHITECTURE = (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
PLATFORM = ROOT / "configs" / "models" / "platform.json"
WEEK_FOUR = [f"M{number}" for number in range(23, 28)]
ROW = re.compile(r"(?m)^\| (M\d+) \| ([^|]+) \| ([^|]+) \| (\d+)/(\d+) \(100%\) \| `(tests/[^`]+)` \((\d+) test")


def table() -> str:
    return TASKS.split("## Bảng tiến độ tuần 4", 1)[1].split("\n\n", 2)[1]


def count_tests(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"))


def vi(value: float) -> str:
    return f"{value:.1f}".replace(".", ",")


def bullets(text: str) -> list[str]:
    return re.findall(r"(?m)^- (.+)$", text)


class TasksTests(unittest.TestCase):
    def test_every_milestone_is_done_with_all_criteria(self):
        rows = [line for line in table().splitlines() if re.match(r"\| M\d+ \|", line)]
        self.assertEqual([re.match(r"\| (M\d+)", row).group(1) for row in rows], WEEK_FOUR)
        for row in rows:
            milestone = re.match(r"\| (M\d+)", row).group(1)
            with self.subTest(milestone):
                self.assertIn("**Xong**", row)
                section = TASKS.split(f"## {milestone}:", 1)[1].split("\n## ", 1)[0]
                self.assertNotIn("- [ ]", section)
                self.assertIn("- [x]", section)
        self.assertRegex(table(), r"(?m)^\| \*\*Tổng\*\* \| 5 mốc \| 5 \*\*Xong\*\* \| 5/5 mốc \(100%\) \|")

    def test_test_counts_in_table_match_test_files(self):
        matches = list(ROW.finditer(table()))
        self.assertEqual([match.group(1) for match in matches], WEEK_FOUR)
        for match in matches:
            path, count = ROOT / match.group(6), int(match.group(7))
            with self.subTest(match.group(1)):
                self.assertTrue(path.is_file(), path)
                self.assertEqual(count_tests(path), count)


class ReadmeTests(unittest.TestCase):
    def test_roadmap_marks_week_four_done_and_status_is_updated(self):
        roadmap = README.split("## Lộ trình tiếp theo", 1)[1]
        for milestone in WEEK_FOUR:
            with self.subTest(milestone): self.assertRegex(roadmap, rf"(?m)^- \*\*{milestone} – [^*]+\*\* \(xong")
        self.assertLess(roadmap.index("**Đề xuất tuần 5**"), roadmap.index("### Tuần 3 (M15–M22, đã xong)"))  # tuần mới ở trên, tuần 3 giữ nguyên bên dưới
        status = README.split("## Trạng thái hiện tại", 1)[1].split("\n## ", 1)[0]
        self.assertIn("Cập nhật 1/10/2026 (tổng kết tuần 4, M27)", status)
        for phrase in ("`max_tokens`", "`vietnamese_aya`", "`medium`", "`train_kaggle` chưa chạy trên Kaggle thật"):
            with self.subTest(phrase): self.assertIn(phrase, status)
        self.assertIn("`TASKS.md` (tuần 4)", README)

    def test_readme_numbers_match_code(self):
        presets = list_presets()
        self.assertEqual(len(presets), 4)
        self.assertIn(f"| Dữ liệu | {len(presets)} preset Hugging Face", README)
        medium = find_model_config(PLATFORM, "medium")
        chosen = load_finetune_config(ROOT / "configs" / "training" / "colab_14b.json").max_length
        status = README.split("## Trạng thái hiện tại", 1)[1].split("\n## ", 1)[0]
        self.assertIn(f"({chosen} token khoảng {vi(training_gb(medium.params_b, '4bit', chosen))} GB)", status)
        self.assertIn(f"dài nhất {max_seq_len(medium.params_b, '4bit', 16)} token", README.split("## Lộ trình tiếp theo", 1)[1])


class MemoryTests(unittest.TestCase):
    def test_week_five_proposals_match_between_readme_and_memory(self):
        readme = bullets(README.split("**Đề xuất tuần 5**", 1)[1].split("\n", 1)[1].split("\n\n", 1)[0])
        memory = bullets(MEMORY.split("## Đề xuất tuần 5\n", 1)[1].split("\n## ", 1)[0])
        self.assertGreaterEqual(len(readme), 3)
        self.assertEqual(readme, memory)

    def test_memory_says_the_week_is_done(self):
        self.assertLessEqual(len(MEMORY), 5000)
        self.assertIn("XONG TUẦN 4", MEMORY)
        checklist = MEMORY.split("## Checklist tuần 4\n", 1)[1].split("\n## ", 1)[0]
        self.assertEqual(re.findall(r"\[x\] (M\d+)", checklist), WEEK_FOUR)
        self.assertNotIn("[ ]", checklist)
        log = MEMORY.split("## Nhật ký tuần 4", 1)[1]
        for milestone in WEEK_FOUR:
            with self.subTest(milestone): self.assertRegex(log, rf"(?m)^\| [^|]+ \| {milestone} \|")


class ArchitectureTests(unittest.TestCase):
    def test_architecture_lists_week_four_parts_and_what_is_still_unverified(self):
        for name in sorted(path.name for path in (ROOT / "configs" / "training").glob("*.json")):
            with self.subTest(name): self.assertIn(name, ARCHITECTURE)
        for name in ("test_m23_max_tokens.py", "test_m24_kaggle.py", "test_m25_model_14b.py", "test_m26_preset_tieng_viet_2.py", "`max_tokens`", "`vietnamese_aya`", "`where`", "`scan_limit`"):
            with self.subTest(name): self.assertIn(name, ARCHITECTURE)
        limits = ARCHITECTURE.split("## An toàn và giới hạn", 1)[1]
        for phrase in ("cập nhật 1/10, tổng kết tuần 4", "`train_kaggle`", "`SEQ_SHARE`", "`vietnamese_aya`", "đề xuất tuần 5"):
            with self.subTest(phrase): self.assertIn(phrase, limits)


if __name__ == "__main__":
    unittest.main()
