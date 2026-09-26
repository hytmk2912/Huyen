"""Mốc M16: notebook bền hơn. Lệnh lỗi thì dừng Run all (hàm `run` của local_ai/colab.py), và chạy lại sau khi Colab ngắt
thì dùng lại báo cáo chấm trước đã lưu trên Hugging Face.

`run` được test với tiến trình thật; Hugging Face Hub là module giả; không cần mạng hay GPU.
"""
import contextlib
import gc
import importlib.machinery
import io
import json
import re
import sys
import tempfile
import types
import unittest
import warnings
from pathlib import Path
from unittest import mock

from local_ai.colab import StepFailed, command_argv, run
from local_ai.evaluation.__main__ import main as evaluation_main

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = {name: ROOT / "notebooks" / f"{name}.ipynb" for name in ("train_colab", "agent_colab")}
REPO = "nguoi-dung/huyen-smoke-qlora"


def code_cells(path: Path) -> dict[str, str]:
    return {cell["id"]: "".join(cell["source"]) for cell in json.loads(path.read_text(encoding="utf-8"))["cells"] if cell["cell_type"] == "code"}


def fake_module(name, **attributes):
    module = types.ModuleType(name); module.__spec__ = importlib.machinery.ModuleSpec(name, None)
    for key, value in attributes.items(): setattr(module, key, value)
    return module


class EntryNotFoundError(Exception): pass
class RepositoryNotFoundError(Exception): pass
class RevisionNotFoundError(Exception): pass


class FakeHub:
    """huggingface_hub giả: giữ file đã đẩy trong một thư mục tạm; hf_hub_download trả file đó, chưa có thì báo EntryNotFoundError."""

    def __init__(self, directory: Path, download_error: Exception | None = None, upload_error: Exception | None = None):
        self.directory, self.download_error, self.upload_error, self.calls = directory, download_error, upload_error, []

    def patch(self):
        hub = self

        def hf_hub_download(repo_id, filename):
            hub.calls.append(("download", filename))
            if hub.download_error: raise hub.download_error
            path = hub.directory / filename
            if not path.is_file(): raise EntryNotFoundError(filename)
            return str(path)

        class HfApi:
            def create_repo(self, **kwargs): hub.calls.append(("create_repo", kwargs["private"]))
            def upload_file(self, path_or_fileobj, path_in_repo, repo_id, commit_message):
                hub.calls.append(("upload", path_in_repo))
                if hub.upload_error: raise hub.upload_error
                target = hub.directory / path_in_repo; target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(Path(path_or_fileobj).read_bytes())

        errors = fake_module("huggingface_hub.errors", EntryNotFoundError=EntryNotFoundError, RepositoryNotFoundError=RepositoryNotFoundError, RevisionNotFoundError=RevisionNotFoundError)
        return mock.patch.dict(sys.modules, {"huggingface_hub": fake_module("huggingface_hub", hf_hub_download=hf_hub_download, HfApi=HfApi, errors=errors), "huggingface_hub.errors": errors})


class RunHelperTests(unittest.TestCase):
    def run_capture(self, command, step="Bước thử", **options):
        output = io.StringIO()
        run(command, step, output=output, **options)
        return output.getvalue()

    def test_streams_output_keeps_vietnamese_and_carriage_return(self):
        printed = self.run_capture(["python", "-c", "import sys; print('Tiếng Việt có dấu'); sys.stdout.write('10%\\r100%\\n'); print(sys.executable)"])
        self.assertIn("Tiếng Việt có dấu\n10%\r100%\n", printed)
        self.assertIn(sys.executable, printed)  # "python" chạy bằng đúng Python của notebook

    def test_failure_raises_with_step_and_exit_code(self):
        with self.assertRaises(StepFailed) as caught:
            self.run_capture("python -c \"import sys; print('dữ liệu hỏng'); sys.exit(3)\"", "Bước 5 (lấy dữ liệu)")
        message = str(caught.exception)
        for phrase in ("Bước 5 (lấy dữ liệu) lỗi", "mã thoát 3", "Các ô sau chưa chạy"):
            with self.subTest(phrase): self.assertIn(phrase, message)
        with self.assertRaisesRegex(StepFailed, "không tìm thấy lệnh 'khong-co-lenh-nay'"): run("khong-co-lenh-nay --version", "Bước 3")

    def test_run_all_stops_at_the_failing_cell(self):
        executed = []
        cells = [("Bước 5", ["python", "-c", "print(1)"]), ("Bước 6", ["python", "-c", "raise SystemExit(2)"]), ("Bước 7", ["python", "-c", "print(3)"])]
        with self.assertRaises(StepFailed):  # Colab dừng Run all ở ô ném lỗi; mô phỏng bằng một vòng lặp qua các ô
            for step, command in cells:
                run(command, step, output=io.StringIO()); executed.append(step)
        self.assertEqual(executed, ["Bước 5"])

    def test_quiet_env_and_no_shell(self):
        self.assertEqual(self.run_capture(["python", "-c", "print('ồn ào')"], quiet=True), "")
        with self.assertRaises(StepFailed):
            output = io.StringIO()
            try: run(["python", "-c", "import sys; print('chi tiết lỗi'); sys.exit(1)"], "Bước 3", quiet=True, output=output)
            finally: self.assertIn("chi tiết lỗi", output.getvalue())  # quiet vẫn in output khi lệnh lỗi
        printed = self.run_capture(["python", "-c", "import os; print(os.environ['OLLAMA_VERSION'], os.environ['PYTHONUNBUFFERED'])"], env={"OLLAMA_VERSION": "0.34.4"})
        self.assertEqual(printed.strip(), "0.34.4 1")
        printed = self.run_capture(["python", "-c", "import sys; print(sys.argv[1])", "a; echo pwned $(id)"])
        self.assertEqual(printed.strip(), "a; echo pwned $(id)")  # không qua shell: ký tự shell chỉ là chữ
        self.assertEqual(command_argv("pip install -q x==1"), [sys.executable, "-m", "pip", "install", "-q", "x==1"])


class RunCleanupTests(unittest.TestCase):
    """Rà soát tuần 3: `run` đóng pipe của tiến trình con và in nốt chữ còn trong decoder."""

    def test_pipe_is_closed_on_success_and_failure(self):
        for label, command in (("chạy xong", ["python", "-c", "print('xong')"]), ("lỗi", ["python", "-c", "raise SystemExit(4)"])):
            with self.subTest(label), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ResourceWarning)
                try: run(command, "Bước thử", output=io.StringIO())
                except StepFailed: pass
                gc.collect()
                self.assertEqual([str(item.message) for item in caught if issubclass(item.category, ResourceWarning)], [])

    def test_flushes_incomplete_utf8_at_the_end(self):
        output = io.StringIO()
        run(["python", "-c", "import sys; sys.stdout.buffer.write('chữ cuối'.encode() + b'\\xe1')"], "Bước thử", output=output)
        self.assertEqual(output.getvalue(), "chữ cuối\ufffd")  # byte cuối bị cắt vẫn được in (thành ký tự thay thế), không mất

class NotebookTests(unittest.TestCase):
    def test_no_shell_escapes_except_first_clone(self):
        for name, path in NOTEBOOKS.items():
            for cell_id, source in code_cells(path).items():
                escapes = [line.strip() for line in source.splitlines() if line.strip().startswith("!")]
                with self.subTest(f"{name}/{cell_id}"):
                    self.assertTrue(all(line.startswith("!git clone") for line in escapes), escapes)  # lúc đó chưa có code repo để gọi run

    def test_every_run_call_names_its_own_step(self):
        for name, path in NOTEBOOKS.items():
            cells = code_cells(path)
            calls = 0
            for cell_id, source in cells.items():
                number = re.match(r"# Bước (\d+):", source).group(1)
                for step in re.findall(r'run\((?:f?"[^"]*"|\[[^\]]*\]), "([^"]+)"', source):
                    calls += 1
                    with self.subTest(f"{name}/{cell_id}: {step}"): self.assertTrue(step.startswith(f"Bước {number} ("), step)
            self.assertGreaterEqual(calls, 6)
            setup = next(source for source in cells.values() if "from local_ai.colab import run" in source)
            self.assertLess(setup.index("%cd /content/Huyen"), setup.index("from local_ai.colab import run"))
            self.assertIn('raise RuntimeError("Tải code thất bại', setup)

    def test_eval_before_training_is_cached_on_hub(self):
        cell = code_cells(NOTEBOOKS["train_colab"])["buoc-7-cham-truoc"]
        self.assertIn("--hub-repo {HUB_REPO} --hub-path eval/truoc", cell)
        # M19 (chủ repo chọn 26/9): chấm sau cũng đẩy báo cáo lên Hub (eval/sau), nhưng vẫn luôn chấm lại vì adapter có thể đã đổi
        self.assertIn("--hub-repo {HUB_REPO} --hub-path eval/sau --no-reuse", code_cells(NOTEBOOKS["train_colab"])["buoc-9-cham-sau"])


class EvalReuseTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(); self.root = Path(self.directory.name)
        models = {"models": [{"name": "may-chu-tat", "backend": "openai_compatible", "base_url": "http://127.0.0.1:9/v1", "source": "gia", "capabilities": ["chat"], "timeout_s": 2, "max_new_tokens": 64}]}
        self.models = self.root / "models.json"; self.models.write_text(json.dumps(models), encoding="utf-8")

    def tearDown(self):
        self.directory.cleanup()

    def evaluate(self, hub, *extra, output="out"):
        argv = ["--models", str(self.models), "--output", str(self.root / output), "--hub-repo", REPO, "--hub-path", "eval/truoc", *extra]
        with hub.patch(), contextlib.redirect_stdout(io.StringIO()) as printed, contextlib.redirect_stderr(io.StringIO()) as error:
            code = evaluation_main(argv)
        return code, printed.getvalue() + error.getvalue()

    def test_first_run_pushes_then_rerun_reuses_without_calling_model(self):
        hub = FakeHub(self.root / "hub")
        code, _ = self.evaluate(hub, "--scripted", output="lan1")
        self.assertEqual(code, 0)
        self.assertEqual([call for call in hub.calls if call[0] == "upload"], [("upload", "eval/truoc/report.json"), ("upload", "eval/truoc/report.md")])
        pushed = json.loads((self.root / "hub" / "eval" / "truoc" / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(set(pushed["settings"]), {"model", "source", "revision", "max_new_tokens", "generation", "cases_sha256"})  # M19 thêm nguồn, revision, cách sinh chữ
        code, printed = self.evaluate(hub, "--scripted", output="lan2")
        self.assertEqual(code, 0); self.assertIn("Dùng lại báo cáo đã chấm trên Hugging Face", printed)
        self.assertEqual(json.loads((self.root / "lan2" / "report.json").read_text(encoding="utf-8")), pushed)
        self.assertEqual(len([call for call in hub.calls if call[0] == "upload"]), 2)  # dùng lại thì không đẩy lại

    def test_reuse_skips_model_and_other_settings_evaluate_again(self):
        hub = FakeHub(self.root / "hub")
        cached = {"status": "completed", "model": "may-chu-tat", "generated_at": "2026-09-25T03:00:00+00:00", "cases": 30, "passed": 7, "accuracy": 0.23, "groups": {}, "languages": {}, "failures": [], "results": []}
        with hub.patch(), contextlib.redirect_stdout(io.StringIO()):
            from local_ai.config.settings import find_model_config
            from local_ai.evaluation.__main__ import eval_settings
            settings = eval_settings(find_model_config(self.models, "may-chu-tat"), ROOT / "data" / "eval" / "eval_v1.jsonl")  # M19: cài đặt tính từ cấu hình model (max_new_tokens 64)
        target = self.root / "hub" / "eval" / "truoc" / "report.json"; target.parent.mkdir(parents=True)
        target.write_text(json.dumps({**cached, "settings": settings}), encoding="utf-8")
        code, printed = self.evaluate(hub, "--model", "may-chu-tat")  # máy chủ model đang tắt: nếu chấm thật thì sẽ lỗi
        self.assertEqual(code, 0); self.assertIn("7/30 câu đạt", printed)
        code, printed = self.evaluate(hub, "--model", "may-chu-tat", "--max-new-tokens", "32", output="khac")
        self.assertEqual(code, 1)  # khác max_new_tokens: chấm lại, và máy chủ tắt nên báo lỗi
        self.assertIn("cài đặt khác", printed)
        del cached["generated_at"]; target.write_text(json.dumps({**cached, "settings": settings}), encoding="utf-8")
        code, printed = self.evaluate(hub, "--model", "may-chu-tat", output="hong")
        self.assertEqual(code, 1); self.assertIn("thiếu thông tin", printed)  # báo cáo trên Hub hỏng: chấm lại, không làm lỗi lệnh

    def test_hub_errors_never_block_evaluation(self):
        code, printed = self.evaluate(FakeHub(self.root / "hub", download_error=OSError("mất mạng")), "--scripted")
        self.assertEqual(code, 0); self.assertIn("sẽ chấm lại", printed)
        code, printed = self.evaluate(FakeHub(self.root / "hub-khac", upload_error=OSError("403")), "--scripted", output="b")
        self.assertEqual(code, 0); self.assertIn("lần chạy lại sẽ phải chấm lại", printed)

    def test_dry_run_shows_hub_plan(self):
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            self.assertEqual(evaluation_main(["--scripted", "--dry-run", "--hub-repo", REPO, "--hub-path", "eval/truoc"]), 0)
        self.assertEqual(json.loads(printed.getvalue())["hub"], {"repo": REPO, "path": "eval/truoc/report.json"})
        with self.assertRaisesRegex(ValueError, "tên-người-dùng/tên-repo"): evaluation_main(["--scripted", "--dry-run", "--hub-repo", "khong-hop-le"])


if __name__ == "__main__":
    unittest.main()
