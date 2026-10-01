"""Mốc M21: tổng kết tuần 3. README, `TASKS.md`, `memory.md` và `docs/ARCHITECTURE.md` khớp thực tế."""
import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Từ khi mở tuần 4 (27/9), kế hoạch tuần 3 chép nguyên văn sang archive/tasks-tuan-3.md và bản cuối memory.md tuần 3 nằm cuối archive/memory-tuan-3.md
# (chủ repo yêu cầu chuyển; chỉ đổi chỗ đọc, giữ nguyên ý kiểm tra).
TASKS = (ROOT / "archive" / "tasks-tuan-3.md").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")
MEMORY = (ROOT / "archive" / "memory-tuan-3.md").read_text(encoding="utf-8").split("## Bản cuối `memory.md` tuần 3", 1)[1].split("````markdown\n", 1)[1].split("\n````", 1)[0] + "\n"
ARCHITECTURE = (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
ROW = re.compile(r"(?m)^\| (M\d+) \| ([^|]+) \| ([^|]+) \| (\d+)/(\d+) \(100%\) \| `(tests/[^`]+)` \((\d+) test")


def table_rows() -> list[tuple[str, ...]]:
    table = TASKS.split("## Bảng tiến độ tuần 3", 1)[1].split("\n\n", 2)[1]
    return [line for line in table.splitlines() if re.match(r"\| M\d+ \|", line)]


def count_tests(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"))


class WeekThreeTests(unittest.TestCase):
    def test_every_chosen_milestone_is_done_with_all_criteria(self):
        rows = table_rows()
        self.assertEqual(sorted(re.match(r"\| (M\d+)", row).group(1) for row in rows), [f"M{number}" for number in range(15, 23)])
        for row in rows:
            milestone = re.match(r"\| (M\d+)", row).group(1)
            with self.subTest(milestone):
                self.assertIn("**Xong**", row)
                section = TASKS.split(f"## {milestone}:", 1)[1].split("\n## ", 1)[0]
                self.assertNotIn("- [ ]", section)
                self.assertIn("- [x]", section)

    def test_test_counts_in_table_match_test_files(self):
        for match in ROW.finditer(TASKS):
            milestone, path, count = match.group(1), ROOT / match.group(6), int(match.group(7))
            with self.subTest(milestone):
                self.assertTrue(path.is_file(), path)
                self.assertEqual(count_tests(path), count)

    def test_readme_roadmap_marks_every_milestone_done(self):
        roadmap = README.split("## Lộ trình tiếp theo", 1)[1]
        for number in range(15, 23):
            with self.subTest(f"M{number}"):
                self.assertRegex(roadmap, rf"\*\*M{number} – [^*]+\*\* \(xong")
        self.assertIn("Chủ repo đã đóng ngày 27/9", roadmap)
        # M27 (tổng kết tuần 4) cập nhật lại ngày ở mục trạng thái: chỉ cần ngày cập nhật không sớm hơn lần tổng kết tuần 3 (27/9/2026).
        day, month, year = map(int, re.search(r"Cập nhật (\d+)/(\d+)/(\d{4})", README.split("## Trạng thái hiện tại", 1)[1].split("\n## ", 1)[0]).groups())
        self.assertGreaterEqual((year, month, day), (2026, 9, 27))

    def test_week_four_proposals_match_between_readme_and_memory(self):
        readme = re.findall(r"(?m)^- (.+)$", README.split("Đề xuất tuần 4", 1)[1].split("\n\n", 1)[0])
        memory = re.findall(r"(?m)^- (.+)$", MEMORY.split("## Đề xuất tuần 4", 1)[1].split("\n## ", 1)[0])
        self.assertGreaterEqual(len(readme), 3)
        self.assertEqual(readme, memory)
        self.assertLessEqual(len(MEMORY), 5000)
        self.assertIn("XONG TUẦN 3", MEMORY)

    def test_architecture_lists_what_is_still_unverified(self):
        limits = ARCHITECTURE.split("## An toàn và giới hạn", 1)[1]
        for phrase in ("Đã chạy thật trên Colab T4", "Chưa kiểm chứng", "agent_trained_colab", "27B", "M19, M20"):
            with self.subTest(phrase): self.assertIn(phrase, limits)


if __name__ == "__main__":
    unittest.main()
