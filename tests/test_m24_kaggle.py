"""Mốc M24: notebook train trên Kaggle (`notebooks/train_kaggle.ipynb`, sinh bằng `notebooks/build.py`).

Không cần mạng, GPU hay Kaggle thật: notebook chỉ được kiểm tra hợp lệ và các lệnh của nó chạy bằng `--dry-run`. Ô đọc token,
ô gom kết quả và lệnh train (khi phiên bị ngắt) chạy với kaggle_secrets, huggingface_hub, torch, trl... giả và thư mục tạm
thay cho /kaggle/working.
"""
import contextlib
import importlib.util
import io
import json
import os
import posixpath
import re
import shlex
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from local_ai.config.settings import find_model_config
from local_ai.data.secrets import SECRET
from local_ai.training import calibrate, finetune

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "train_kaggle.ipynb"
COLAB_NOTEBOOK = ROOT / "notebooks" / "train_colab.ipynb"
PLATFORM = ROOT / "configs" / "models" / "platform.json"
FIXTURES = ROOT / "tests" / "fixtures" / "measurements"
WORKING = "/kaggle/working"
CODE_DIR = "/kaggle/working/Huyen"
KAGGLE_BUTTON = "[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/kernels/welcome?src=https://github.com/hytmk2912/Huyen/blob/main/notebooks/train_kaggle.ipynb)"
STEPS = ["gioi-thieu", "buoc-1-chon-model", "buoc-2-gpu", "buoc-3-cai-dat", "buoc-4-token", "buoc-5-du-lieu", "buoc-6-uoc-tinh", "buoc-7-cham-truoc", "buoc-8-train",
         "buoc-9-cham-sau", "buoc-10-so-sanh", "buoc-11-day-adapter", "buoc-12-so-do", "buoc-13-gom-ket-qua", "ket-qua"]
VIETNAMESE = re.compile(r"[ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]", re.I)
COMMAND = re.compile(r'run\(f?"(python -m local_ai[^"]*)"')
OUTPUT_FLAGS = ("--output", "--output-dir", "--adapter-dir", "--eval-before", "--eval-after", "--train-data")
KAGGLE_ONLY = " --output .runs/so_do/kaggle-{MODEL}.json", " --hub-path so_do/kaggle-{MODEL}.json"  # phần lệnh số đo chỉ Kaggle có


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


BUILD = load_module("build_notebooks_m24", ROOT / "notebooks" / "build.py")
M10 = load_module("m10_helpers_m24", ROOT / "tests" / "test_m10_colab.py")  # huggingface_hub giả và module giả


def code_cells(path: Path = NOTEBOOK) -> dict[str, str]:
    return {cell["id"]: "".join(cell["source"]) for cell in json.loads(path.read_text(encoding="utf-8"))["cells"] if cell["cell_type"] == "code"}


def variables(model: str) -> dict[str, str]:
    """Chạy ô Bước 1 (chỉ có Python thường) với MODEL đã chọn; HUB_REPO là tên repo mẫu."""
    source = re.sub(r'^MODEL = "\w+"', f'MODEL = "{model}"', code_cells()["buoc-1-chon-model"], flags=re.M)
    namespace: dict = {}
    with contextlib.redirect_stdout(io.StringIO()): exec(source, namespace)
    return {"MODEL": namespace["MODEL"], "MAX_NEW_TOKENS": str(namespace["MAX_NEW_TOKENS"]), "HUB_REPO": f"nguoi-dung/huyen-{model}-qlora"}


def templates(path: Path = NOTEBOOK) -> list[str]:
    return [command for source in code_cells(path).values() for command in COMMAND.findall(source)]


def commands(model: str) -> list[list[str]]:
    values = variables(model)
    return [shlex.split(re.sub(r"\{(\w+)\}", lambda match: values[match.group(1)], command)) for command in templates()]


def option(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv else None


class NotebookFileTests(unittest.TestCase):
    def test_valid_nbformat_matches_builder_and_has_no_outputs(self):
        if importlib.util.find_spec("nbformat") is None: self.skipTest("thiếu nbformat (python -m pip install nbformat)")
        import nbformat
        nbformat.validate(nbformat.read(str(NOTEBOOK), as_version=4))
        self.assertIs(BUILD.NOTEBOOKS["train_kaggle.ipynb"], BUILD.train_kaggle)
        self.assertEqual(NOTEBOOK.read_text(encoding="utf-8"), BUILD.render(BUILD.train_kaggle), "file .ipynb khác notebooks/build.py: hãy chạy python notebooks/build.py")
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual([cell["id"] for cell in notebook["cells"]], STEPS)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                with self.subTest(cell["id"]): self.assertEqual((cell["outputs"], cell["execution_count"]), ([], None))
        self.assertEqual((notebook["metadata"]["kaggle"]["accelerator"], notebook["metadata"]["kaggle"]["isInternetEnabled"]), ("nvidiaTeslaT4", True))
        self.assertNotIn("colab", notebook["metadata"])

    def test_every_code_cell_starts_with_vietnamese_comment(self):
        for number, (cell_id, source) in enumerate(code_cells().items(), start=1):
            with self.subTest(cell_id):
                first = source.splitlines()[0]
                self.assertTrue(first.startswith(f"# Bước {number}:"), first)
                self.assertRegex(first, VIETNAMESE)

    def test_library_versions_are_pinned_like_colab(self):
        install = [re.search(r'"pip install -q ([^"]+)"', source) for source in (code_cells()["buoc-3-cai-dat"], code_cells(COLAB_NOTEBOOK)["buoc-3-cai-dat"])]
        packages = install[0].group(1).split()
        self.assertEqual(install[0].group(1), BUILD.PINNED)
        self.assertEqual(install[0].group(1), install[1].group(1))  # cùng bản thư viện đã chạy thật trên Colab
        self.assertTrue(packages and all(re.fullmatch(r"[a-z][a-z0-9_-]*==[0-9][0-9.]*", name) for name in packages), packages)
        self.assertNotIn("torch", [name.split("==")[0] for name in packages])  # Kaggle có sẵn torch hợp với GPU, không cài lại
        self.assertIn('run("pip uninstall -y -q torchao", "Bước 3 (gỡ torchao)")', code_cells()["buoc-3-cai-dat"])

    def test_commands_run_through_run_and_name_their_step(self):
        calls = 0
        for cell_id, source in code_cells().items():
            number = re.match(r"# Bước (\d+):", source).group(1)
            escapes = [line.strip() for line in source.splitlines() if line.strip().startswith("!")]
            with self.subTest(cell_id): self.assertTrue(all(line.startswith("!git clone") for line in escapes), escapes)
            for step in re.findall(r'run\((?:f?"[^"]*"|\[[^\]]*\]), "([^"]+)"', source):
                calls += 1
                with self.subTest(f"{cell_id}: {step}"): self.assertTrue(step.startswith(f"Bước {number} ("), step)
        self.assertGreaterEqual(calls, 11)


class KaggleEnvironmentTests(unittest.TestCase):
    def run_token_cell(self, get_secret):
        cell = code_cells()["buoc-4-token"]
        client = type("UserSecretsClient", (), {"get_secret": lambda self, name: get_secret(name)})
        modules = {"kaggle_secrets": M10.fake_module("kaggle_secrets", UserSecretsClient=client), "huggingface_hub": M10.fake_module("huggingface_hub", whoami=lambda: {"name": "nguoi-dung"})}
        namespace = {"MODEL": "light"}
        with mock.patch.dict(sys.modules, modules), mock.patch.dict(os.environ, {}), contextlib.redirect_stdout(io.StringIO()) as printed:
            os.environ.pop("HF_TOKEN", None)
            try: exec(cell, namespace)
            finally: token = os.environ.get("HF_TOKEN")
        return namespace, token, printed.getvalue()

    def test_token_is_read_from_kaggle_secrets(self):
        asked = []
        namespace, token, printed = self.run_token_cell(lambda name: asked.append(name) or "token-gia-khong-that")
        self.assertEqual((asked, token, namespace["HUB_REPO"]), (["HF_TOKEN"], "token-gia-khong-that", "nguoi-dung/huyen-light-qlora"))
        self.assertNotIn("token-gia-khong-that", printed)  # token không bị in ra

    def test_missing_secret_is_explained_in_vietnamese(self):
        def missing(name): raise ConnectionError("không có secret")
        with self.assertRaisesRegex(RuntimeError, r"Chưa đọc được HF_TOKEN: vào Add-ons → Secrets.*\(ConnectionError\)"): self.run_token_cell(missing)

    def test_no_colab_secret_or_token_in_notebook(self):
        text = NOTEBOOK.read_text(encoding="utf-8")
        for colab_only in ("google.colab", "userdata", "/content"):
            with self.subTest(colab_only): self.assertNotIn(colab_only, text)
        self.assertIsNone(SECRET.search(text))
        self.assertFalse([line for source in code_cells().values() for line in source.splitlines() if "print(" in line and "HF_TOKEN" in line])

    def test_only_one_gpu_is_used(self):
        cell = code_cells()["buoc-2-gpu"]
        self.assertLess(cell.index('os.environ["CUDA_VISIBLE_DEVICES"] = "0"'), cell.index("import torch"))  # đặt trước khi torch khởi tạo CUDA
        self.assertIn("torch.cuda.is_available()", cell)

    def test_code_and_outputs_live_under_kaggle_working(self):
        setup = code_cells()["buoc-3-cai-dat"]
        self.assertIn(f"!git clone --depth 1 {BUILD.REPO_URL} {CODE_DIR}", setup)
        self.assertLess(setup.index(f"%cd {CODE_DIR}"), setup.index("from local_ai.colab import run"))
        self.assertIn('raise RuntimeError("Tải code thất bại', setup)
        checked = 0
        for model in ("smoke", "light"):
            for argv in commands(model):
                paths = [option(argv, flag) for flag in OUTPUT_FLAGS if option(argv, flag)] + ([argv[3], argv[4]] if argv[2] == "local_ai.evaluation.compare" else [])
                for path in paths:
                    checked += 1
                    with self.subTest(model=model, path=path):
                        self.assertTrue(posixpath.normpath(posixpath.join(CODE_DIR, path)).startswith(WORKING + "/"), path)  # đường dẫn tương đối nằm dưới /kaggle/working/Huyen
            output_dir = finetune.load_finetune_config(ROOT / "configs" / "training" / f"colab_{model}.json").output_dir
            self.assertTrue(posixpath.normpath(posixpath.join(CODE_DIR, output_dir)).startswith(WORKING + "/"), output_dir)  # checkpoint và adapter
            self.assertTrue(find_model_config(PLATFORM, f"{model}-colab").adapter_path.startswith(output_dir + "/"))
        self.assertGreaterEqual(checked, 20)

    def test_result_cell_copies_results_into_kaggle_working(self):
        cell = code_cells()["buoc-13-gom-ket-qua"]
        self.assertIn('Path("/kaggle/working/ket_qua") / MODEL', cell)
        with tempfile.TemporaryDirectory() as directory:
            working, previous = Path(directory) / "working", os.getcwd()
            code_dir = working / "Huyen"
            for name, text in {".runs/colab_smoke/adapter/adapter_config.json": "{}", ".runs/eval/smoke/truoc/report.json": "{}", ".runs/eval/smoke/sau/report.md": "bảng",
                               ".runs/colab_smoke/measurements.json": "{}"}.items():
                path = code_dir / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text, encoding="utf-8")
            os.chdir(code_dir)
            try:
                with contextlib.redirect_stdout(io.StringIO()) as printed: exec(cell.replace(WORKING, str(working)), {"MODEL": "smoke"})
            finally: os.chdir(previous)
            result = working / "ket_qua" / "smoke"
            copied = sorted(path.relative_to(result).as_posix() for path in result.rglob("*") if path.is_file())
        self.assertEqual(copied, ["adapter/adapter_config.json", "eval/sau/report.md", "eval/truoc/report.json", "measurements.json"])
        self.assertIn("Chưa có .runs/so_do/kaggle-smoke.json - bỏ qua", printed.getvalue())  # chưa có số đo thì bỏ qua, không lỗi


class CommandTests(unittest.TestCase):
    def test_commands_run_with_dry_run_for_both_models(self):
        environ = {**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONPATH": str(ROOT)}
        for model in ("smoke", "light"):
            outputs = []
            for argv in commands(model):
                with self.subTest(" ".join(argv)):
                    result = subprocess.run([sys.executable, *argv[1:], "--dry-run"], cwd=ROOT, env=environ, capture_output=True, text=True, timeout=120)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    try: outputs.append((argv[2], json.loads(result.stdout)))
                    except json.JSONDecodeError: outputs.append((argv[2], result.stdout))
            with self.subTest(model):
                self.assertEqual([module for module, _ in outputs], ["local_ai.data", "local_ai.training.estimate", "local_ai.evaluation", "local_ai.training.finetune", "local_ai.evaluation",
                                                                     "local_ai.evaluation.compare", "local_ai.training.hub", "local_ai.training.calibrate"])
                (_, data), _, (_, before), (_, train), (_, after), _, (_, push), (_, measure) = outputs
                hub_repo = f"nguoi-dung/huyen-{model}-qlora"
                self.assertEqual(sum(item["rows"] for item in data["presets"].values()) + data["tool_calls"]["rows"], 2000)
                self.assertEqual((before["model"]["name"], after["model"]["name"]), (model, f"{model}-colab"))
                self.assertEqual((train["qlora"], train["base_model"]["dtype"]), (True, "float16"))
                self.assertEqual(train["hub"], {"push_to_hub": True, "hub_model_id": hub_repo, "private": True, "hub_strategy": "checkpoint", "resume_from_hub": True})  # phiên bị ngắt thì tải last-checkpoint
                self.assertEqual((push["repo_id"], push["adapter_dir"]), (hub_repo, after["model"]["adapter_path"]))
                self.assertEqual(push["adapter_dir"], train["output_dir"] + "/adapter")
                self.assertEqual((measure["output"], measure["push_to_hub"], measure["hub_path"]), (f".runs/so_do/kaggle-{model}.json", hub_repo, f"so_do/kaggle-{model}.json"))

    def test_same_commands_as_colab_except_measurement_path(self):
        kaggle, colab = templates(), templates(COLAB_NOTEBOOK)
        self.assertEqual(len(kaggle), len(colab))
        for kaggle_command, colab_command in zip(kaggle, colab):
            with self.subTest(colab_command):
                if "local_ai.training.calibrate" in kaggle_command:
                    for extra in KAGGLE_ONLY: self.assertIn(extra, kaggle_command); kaggle_command = kaggle_command.replace(extra, "")
                self.assertEqual(kaggle_command, colab_command)
        self.assertEqual(re.search(r'HUB_REPO = (f"[^"]+")', code_cells()["buoc-4-token"]).group(1), re.search(r'HUB_REPO = (f"[^"]+")', code_cells(COLAB_NOTEBOOK)["buoc-4-token"]).group(1))  # chung repo Hub


class ResumeAfterSessionCutTests(unittest.TestCase):
    """Chạy đúng lệnh train của notebook (thêm --dataset-path, --output-dir tạm) với thư viện giả: phiên mới có /kaggle/working trống."""

    def train(self, hub_files=None, hub_error=None):
        calls = []

        class Trainer:
            def __init__(self, **kwargs): calls.append(("SFTTrainer", kwargs))
            def train(self, resume_from_checkpoint): calls.append(("train", resume_from_checkpoint)); return types.SimpleNamespace(metrics={"train_loss": 1.0})
            def save_model(self, path): calls.append(("save_model", path))

        class Auto:
            def __init__(self, result=None): self.result = result
            def from_pretrained(self, source, **kwargs): return self.result or types.SimpleNamespace(config=types.SimpleNamespace())

        tokenizer = types.SimpleNamespace(pad_token=None, eos_token="<eos>", save_pretrained=lambda path: None)
        dataset = types.SimpleNamespace(select_columns=lambda columns: dataset)
        modules = {
            "torch": M10.fake_module("torch", bfloat16="bf16", float16="fp16", float32="fp32", cuda=types.SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: False)),
            "transformers": M10.fake_module("transformers", AutoTokenizer=Auto(tokenizer), AutoModelForCausalLM=Auto(), BitsAndBytesConfig=lambda **kwargs: kwargs),
            "bitsandbytes": M10.fake_module("bitsandbytes"),
            "trl": M10.fake_module("trl", SFTConfig=lambda **kwargs: kwargs, SFTTrainer=Trainer),
            "peft": M10.fake_module("peft", LoraConfig=lambda **kwargs: kwargs, prepare_model_for_kbit_training=lambda model, **kwargs: model),
            "datasets": M10.fake_module("datasets", load_dataset=lambda *args, **kwargs: dataset)}
        argv = next(argv for argv in commands("smoke") if argv[2] == "local_ai.training.finetune")
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "sft.jsonl"; data.write_text('{"messages": []}\n', encoding="utf-8")
            output = Path(directory) / "working" / "Huyen" / ".runs" / "colab_smoke"  # phiên mới: chưa có checkpoint nào trong máy
            with mock.patch.dict(sys.modules, modules), M10.fake_hub(calls, files=hub_files, error=hub_error), contextlib.redirect_stdout(io.StringIO()) as printed, contextlib.redirect_stderr(io.StringIO()):
                finetune.main([*argv[3:], "--dataset-path", str(data), "--output-dir", str(output)])
            relative = lambda value: Path(value).relative_to(output).as_posix() if isinstance(value, str) and value.startswith(str(output)) else value
            return json.loads(printed.getvalue()), [(name, relative(value)) for name, value in calls]

    def test_first_session_trains_from_scratch_and_pushes_checkpoints(self):
        result, calls = self.train(hub_error=M10.RepositoryNotFoundError("chưa có repo"))
        self.assertEqual(result["status"], "completed")
        self.assertIn(("train", None), calls)
        args = next(value for name, value in calls if name == "SFTTrainer")["args"]
        self.assertEqual({key: args[key] for key in ("push_to_hub", "hub_model_id", "hub_strategy", "hub_private_repo", "fp16")},
                         {"push_to_hub": True, "hub_model_id": "nguoi-dung/huyen-smoke-qlora", "hub_strategy": "checkpoint", "hub_private_repo": True, "fp16": True})

    def test_session_cut_resumes_from_hub_checkpoint(self):
        _, calls = self.train(hub_files={"last-checkpoint/trainer_state.json": '{"global_step": 50}', "last-checkpoint/adapter_model.safetensors": "x"})
        downloads = [value for name, value in calls if name == "snapshot_download"]
        self.assertEqual([(item["repo_id"], item["allow_patterns"]) for item in downloads], [("nguoi-dung/huyen-smoke-qlora", ["last-checkpoint/*"])])
        self.assertIn(("train", "_hub/last-checkpoint"), calls)  # train tiếp từ checkpoint vừa tải từ Hub


class CalibrateHubPathTests(unittest.TestCase):
    def push(self, *extra):
        calls = []

        class HfApi:
            def create_repo(self, **kwargs): calls.append(("create_repo", kwargs))
            def upload_file(self, **kwargs): calls.append(("upload_file", kwargs))

        with tempfile.TemporaryDirectory() as directory:
            argv = ["--config", str(ROOT / "configs" / "training" / "colab_smoke.json"), "--measurements", str(FIXTURES / "smoke_t4.json"), "--output", str(Path(directory) / "so_do.json"),
                    "--push-to-hub", "--hub-model-id", "nguoi-dung/huyen-smoke-qlora", *extra]
            with mock.patch.dict(sys.modules, {"huggingface_hub": M10.fake_module("huggingface_hub", HfApi=HfApi)}), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(calibrate.main(argv), 0)
        return [value["path_in_repo"] for name, value in calls if name == "upload_file"]

    def test_kaggle_measurements_do_not_overwrite_colab_measurements(self):
        self.assertEqual(self.push(), ["so_do/smoke.json"])  # Colab: giữ đường dẫn cũ
        self.assertEqual(self.push("--hub-path", "so_do/kaggle-smoke.json"), ["so_do/kaggle-smoke.json"])

    def test_invalid_hub_path_is_rejected_in_vietnamese(self):
        for path in ("../so_do/smoke.json", "/so_do/smoke.json", "so_do/smoke.txt"):
            with self.subTest(path), contextlib.redirect_stderr(io.StringIO()) as error, self.assertRaises(SystemExit):
                calibrate.main(["--config", str(ROOT / "configs" / "training" / "colab_smoke.json"), "--hub-path", path, "--dry-run"])
            self.assertIn("--hub-path phải là đường dẫn tương đối trong repo", error.getvalue())


class ReadmeTests(unittest.TestCase):
    def test_readme_has_kaggle_button_and_iphone_guide(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        section = readme.split("## Train trên Kaggle miễn phí\n", 1)[1].split("\n## ", 1)[0]
        self.assertTrue(section.startswith(KAGGLE_BUTTON))
        guide = section.split("### Hướng dẫn chạy trên iPhone\n", 1)[1]
        for phrase in ("Safari", "Open In Kaggle", "Yêu cầu trang web cho máy tính", "Phone Verification", "GPU T4 x2", "Internet → On", "Add-ons → Secrets", "`HF_TOKEN`",
                       'MODEL = "light"', "Save & Run All (Commit)", "ket_qua/<model>", "**Khi phiên bị ngắt**", "last-checkpoint", "Lỗi hay gặp trên Kaggle"):
            with self.subTest(phrase): self.assertIn(phrase, guide)
        for phrase in ("/kaggle/working", "chưa chạy thử trên Kaggle thật", "so_do/kaggle-<model>.json", "tests/test_m24_kaggle.py"):
            with self.subTest(phrase): self.assertIn(phrase, section)
        intro = "".join(json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"][0]["source"])
        self.assertIn('mục "Train trên Kaggle miễn phí" trong [README]', intro)


if __name__ == "__main__":
    unittest.main()
