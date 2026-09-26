"""Lượt rà soát tuần 3 (26/9, chủ repo yêu cầu, không phải mốc).

- Lệnh `python -m local_ai.check`: test + compileall + secret-scan; test bị bỏ qua vì thiếu thư viện tính là chưa xanh,
  trừ khi có `--allow-skip`.
- `TASKS.md` chỉ giữ tuần 3 (tuần 1–2 chép nguyên văn sang `archive/tasks-tuan-1-2.md`); `memory.md` tối đa 5.000 ký tự.
- Skill `lam-moc` gọi đúng tên mục có trong `memory.md`; bảng tiến độ tuần 3 có dòng "Tổng".
- README khớp thực tế sau khi chạy thật trên Colab T4 (25–26/9).
"""
import contextlib
import io
import re
import sys
import tempfile
import unittest
from pathlib import Path

from local_ai import check
from local_ai.training.estimate import GPUS

ROOT = Path(__file__).resolve().parent.parent
TASKS = (ROOT / "TASKS.md").read_text(encoding="utf-8")
MEMORY = (ROOT / "memory.md").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")
SKILL = (ROOT / ".claude" / "skills" / "lam-moc" / "SKILL.md").read_text(encoding="utf-8")
CLAUDE = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")


class CheckCommandTests(unittest.TestCase):
    def run_with_tests(self, source: str, allow_skip: bool = False) -> tuple[int, str]:
        with tempfile.TemporaryDirectory() as directory:
            name = f"test_tam_{abs(hash((source, allow_skip)))}"  # tên module riêng cho mỗi lần, không đụng module đã nạp
            (Path(directory) / f"{name}.py").write_text(source, encoding="utf-8")
            output = io.StringIO()
            try:
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
                    code = check.run_checks(Path(directory), allow_skip=allow_skip)
            finally:
                sys.modules.pop(name, None)
                if directory in sys.path: sys.path.remove(directory)
        return code, output.getvalue()

    def test_green_when_everything_passes(self):
        code, printed = self.run_with_tests("import unittest\nclass T(unittest.TestCase):\n    def test_ok(self): pass\n")
        self.assertEqual(code, 0)
        for phrase in ("1. Test: 1 test, 0 sai, 0 lỗi, 0 bị bỏ qua", "2. compileall local_ai: sạch", "3. secret-scan: không có phát hiện", "KẾT QUẢ: XANH"):
            with self.subTest(phrase): self.assertIn(phrase, printed)

    def test_skip_for_missing_library_is_not_green_unless_allowed(self):
        source = "import unittest\nclass T(unittest.TestCase):\n    @unittest.skip('thiếu thư viện: torch')\n    def test_train(self): pass\n"
        code, printed = self.run_with_tests(source)
        self.assertEqual(code, 1)
        self.assertIn("có test bị bỏ qua vì thiếu thư viện", printed); self.assertIn("torch bản CPU", printed)
        code, printed = self.run_with_tests(source, allow_skip=True)
        self.assertEqual(code, 0); self.assertIn("--allow-skip", printed)

    def test_failing_test_is_not_green(self):
        code, printed = self.run_with_tests("import unittest\nclass T(unittest.TestCase):\n    def test_sai(self): self.assertEqual(1, 2)\n")
        self.assertEqual(code, 1); self.assertIn("có test sai hoặc lỗi", printed)

    def test_library_skip_detection(self):
        skipped = [("a", "thiếu nbformat (python -m pip install nbformat)"), ("b", "thiếu thư viện hoặc transformers chưa có Qwen3.5: "), ("c", "chỉ chạy trên Windows")]
        self.assertEqual([name for name, _ in check.library_skips(skipped)], ["a", "b"])

    def test_rules_require_the_check_before_done_and_merge(self):
        for text in (CLAUDE, SKILL, README):
            with self.subTest(text[:30]): self.assertIn("python -m local_ai.check", text)
        self.assertIn("--allow-skip", SKILL); self.assertIn("torch bản CPU", CLAUDE)


class ShorterDocsTests(unittest.TestCase):
    def test_tasks_keeps_only_week_three(self):
        archive = (ROOT / "archive" / "tasks-tuan-1-2.md").read_text(encoding="utf-8")
        self.assertNotIn("# Nhiệm vụ tuần 2 (M8–M14)", TASKS); self.assertNotIn("# Nhiệm vụ tuần 1", TASKS)
        self.assertIn("archive/tasks-tuan-1-2.md", TASKS)
        self.assertIn("# Nhiệm vụ tuần 2 (M8–M14)", archive); self.assertIn("# Nhiệm vụ tuần 1", archive)
        for milestone in range(1, 15):
            with self.subTest(f"M{milestone}"): self.assertRegex(archive, rf"\| M{milestone} \|")

    def test_memory_is_short_and_skill_uses_its_sections(self):
        self.assertLessEqual(len(MEMORY), 5000)
        self.assertIn("archive/memory-tuan-3.md", MEMORY)
        sections = re.findall(r"(?m)^## (.+)$", MEMORY)
        for name in ("Lỗi còn tồn", "Nhật ký tuần 3", "Việc dở", "Mốc đang làm"):
            with self.subTest(name):
                self.assertIn(name, sections); self.assertIn(f'"{name}"', SKILL)
        self.assertNotIn('"Lỗi gặp"', SKILL); self.assertNotIn('"Nhật ký các lượt"', SKILL)
        for text in (CLAUDE, SKILL):
            with self.subTest(text[:30]): self.assertIn("5.000 ký tự", text); self.assertNotIn("80 dòng", text)

    def test_week_three_table_has_total_row(self):
        table = TASKS.split("## Bảng tiến độ tuần 3", 1)[1].split("\n\n", 2)[1]
        rows = re.findall(r"(?m)^\| (M\d+) \| [^|]+ \| ([^|]+) \|", table)
        done = [milestone for milestone, status in rows if status.strip() == "**Xong**"]
        self.assertRegex(table, rf"(?m)^\| \*\*Tổng\*\* \| .*{len(done)}/{len(rows)} mốc")


class ReadmeFactsTests(unittest.TestCase):
    def test_measured_facts_match_code(self):
        self.assertNotIn("5,4 TFLOPS", README)
        self.assertIn(f"{GPUS['T4'].train_tflops:g} TFLOPS".replace(".", ","), README)
        self.assertIn("tuần 1–3", README); self.assertNotIn("Kế hoạch tuần 1–2", README)
        self.assertIn("PR #4", README); self.assertIn("đề nghị đóng", README)

    def test_no_outdated_not_run_claims(self):
        for outdated in ("**Notebook chưa chạy thử trên Colab thật.**", "**Chưa chạy với Ollama thật**", "**Các lệnh train trên GPU chưa chạy thử**",
                         "`light` chưa chạy thử trên Colab thật", "mới test bằng module giả**, chưa chạy trên GPU"):
            with self.subTest(outdated): self.assertNotIn(outdated, README)
        status = README.split("## Trạng thái hiện tại", 1)[1].split("\n## ", 1)[0]
        self.assertIn("Colab T4", status)


if __name__ == "__main__":
    unittest.main()
