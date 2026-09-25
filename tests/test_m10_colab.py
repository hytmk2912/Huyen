"""Mốc M10: notebook train trên Colab miễn phí (model smoke 0.5B, QLoRA fp16), đẩy checkpoint và adapter lên Hugging Face.

Không cần mạng, GPU hay model thật: notebook chỉ được kiểm tra hợp lệ, các lệnh của nó chạy bằng `--dry-run`,
còn huggingface_hub, torch, transformers, trl, peft và datasets đều là module giả.
"""
import contextlib
import importlib.machinery
import importlib.util
import io
import json
import os
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
from local_ai.evaluation.compare import compare_reports, main as compare_main
from local_ai.training import finetune, hub

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "train_colab.ipynb"
PLATFORM = ROOT / "configs" / "models" / "platform.json"
COLAB_CONFIG = ROOT / "configs" / "training" / "colab_smoke.json"
COLAB_LINK = "https://colab.research.google.com/github/hytmk2912/Huyen/blob/main/notebooks/train_colab.ipynb"
HUB_REPO = "nguoi-dung/huyen-smoke-qlora"
# M11 thêm ô chọn model (buoc-1-chon-model) và ô ước tính thời gian (buoc-6-uoc-tinh); M15 thêm ô số đo thật (buoc-12-so-do).
STEPS = ["gioi-thieu", "buoc-1-chon-model", "buoc-2-gpu", "buoc-3-cai-dat", "buoc-4-token", "buoc-5-du-lieu", "buoc-6-uoc-tinh", "buoc-7-cham-truoc", "buoc-8-train", "buoc-9-cham-sau", "buoc-10-so-sanh", "buoc-11-day-adapter", "buoc-12-so-do", "ket-qua"]
VIETNAMESE = re.compile(r"[ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]", re.I)


def load_notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def code_cells(notebook: dict) -> dict[str, str]:
    return {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"}


def notebook_variables(model: str = "smoke") -> dict[str, str]:
    """Biến mà các lệnh `!python` dùng: chạy ô chọn model (chỉ có Python thường) với MODEL đã chọn; HUB_REPO là tên repo mẫu."""
    source = re.sub(r'^MODEL = "\w+"', f'MODEL = "{model}"', code_cells(load_notebook())["buoc-1-chon-model"], flags=re.M)
    namespace: dict = {}
    with contextlib.redirect_stdout(io.StringIO()): exec(source, namespace)
    return {"MODEL": namespace["MODEL"], "MAX_NEW_TOKENS": str(namespace["MAX_NEW_TOKENS"]), "HUB_REPO": f"nguoi-dung/huyen-{model}-qlora"}


def notebook_commands(model: str = "smoke") -> list[list[str]]:
    """Mọi lệnh `!python -m local_ai...` trong notebook, thay các biến {TÊN} bằng giá trị ứng với model đã chọn."""
    variables = notebook_variables(model)
    lines = [line.strip() for source in code_cells(load_notebook()).values() for line in source.splitlines()]
    return [shlex.split(re.sub(r"\{(\w+)\}", lambda match: variables[match.group(1)], line[1:])) for line in lines if line.startswith("!python -m local_ai")]


def run_notebook_commands(test: unittest.TestCase, model: str = "smoke") -> list[tuple[str, object]]:
    """Chạy mọi lệnh của notebook với --dry-run (không mạng, không GPU); trả về (module, kết quả JSON hoặc chữ nếu lệnh in chữ)."""
    environ = {**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONPATH": str(ROOT)}
    outputs = []
    for command in notebook_commands(model):
        with test.subTest(" ".join(command)):
            result = subprocess.run([sys.executable, *command[1:], "--dry-run"], cwd=ROOT, env=environ, capture_output=True, text=True, timeout=120)
            test.assertEqual(result.returncode, 0, result.stderr)
            try: outputs.append((command[2], json.loads(result.stdout)))
            except json.JSONDecodeError: outputs.append((command[2], result.stdout))
    return outputs


def fake_module(name, **attributes):
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, None)
    for key, value in attributes.items(): setattr(module, key, value)
    return module


class RepositoryNotFoundError(Exception): pass
class RevisionNotFoundError(Exception): pass
class EntryNotFoundError(Exception): pass


def fake_hub(calls, files=None, error=None):
    """huggingface_hub giả: snapshot_download ghi `files` (đường dẫn tương đối -> nội dung) vào local_dir hoặc ném `error`."""
    def snapshot_download(**kwargs):
        calls.append(("snapshot_download", kwargs))
        if error: raise error
        for name, text in (files or {}).items():
            path = Path(kwargs["local_dir"]) / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text, encoding="utf-8")
        return kwargs["local_dir"]

    class HfApi:
        def create_repo(self, **kwargs): calls.append(("create_repo", kwargs))
        def upload_folder(self, **kwargs): calls.append(("upload_folder", kwargs))

    errors = fake_module("huggingface_hub.errors", RepositoryNotFoundError=RepositoryNotFoundError, RevisionNotFoundError=RevisionNotFoundError, EntryNotFoundError=EntryNotFoundError)
    return mock.patch.dict(sys.modules, {"huggingface_hub": fake_module("huggingface_hub", snapshot_download=snapshot_download, HfApi=HfApi, errors=errors), "huggingface_hub.errors": errors})


class NotebookTests(unittest.TestCase):
    def test_notebook_is_valid_nbformat(self):
        if importlib.util.find_spec("nbformat") is None: self.skipTest("thiếu nbformat (python -m pip install nbformat)")
        import nbformat
        nbformat.validate(nbformat.read(str(NOTEBOOK), as_version=4))

    def test_notebook_matches_builder_and_has_no_outputs(self):
        spec = importlib.util.spec_from_file_location("build_notebooks", ROOT / "notebooks" / "build.py")
        build = importlib.util.module_from_spec(spec); spec.loader.exec_module(build)
        self.assertEqual(NOTEBOOK.read_text(encoding="utf-8"), build.render(build.train_colab), "file .ipynb khác notebooks/build.py: hãy chạy python notebooks/build.py")
        notebook = load_notebook()
        self.assertEqual((notebook["nbformat"], notebook["metadata"]["accelerator"], notebook["metadata"]["colab"]["gpuType"]), (4, "GPU", "T4"))
        self.assertEqual([cell["id"] for cell in notebook["cells"]], STEPS)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                with self.subTest(cell["id"]): self.assertEqual((cell["outputs"], cell["execution_count"]), ([], None))

    def test_every_code_cell_starts_with_vietnamese_comment(self):
        for number, (cell_id, source) in enumerate(code_cells(load_notebook()).items(), start=1):
            with self.subTest(cell_id):
                first = source.splitlines()[0]
                self.assertTrue(first.startswith(f"# Bước {number}:"), first)
                self.assertRegex(first, VIETNAMESE)

    def test_gpu_install_and_token_cells(self):
        cells = code_cells(load_notebook())
        self.assertIn("torch.cuda.is_available()", cells["buoc-2-gpu"]); self.assertIn("T4", cells["buoc-2-gpu"])
        install = next(line for line in cells["buoc-3-cai-dat"].splitlines() if "pip install" in line)
        packages = install.split("pip install", 1)[1].split()[1:]  # bỏ cờ -q
        self.assertTrue(packages and all(re.fullmatch(r"[a-z][a-z0-9_-]*==[0-9][0-9.]*", name) for name in packages), packages)
        self.assertNotIn("torch", [name.split("==")[0] for name in packages])  # Colab có sẵn torch hợp với GPU, không cài lại
        self.assertIn('userdata.get("HF_TOKEN")', cells["buoc-4-token"])
        text = NOTEBOOK.read_text(encoding="utf-8")
        self.assertIsNone(SECRET.search(text))
        self.assertFalse([line for source in cells.values() for line in source.splitlines() if "print(" in line and "HF_TOKEN" in line])

    def test_notebook_commands_run_with_dry_run(self):
        outputs = run_notebook_commands(self)  # model mặc định của notebook: smoke
        self.assertEqual([module for module, _ in outputs], ["local_ai.data", "local_ai.training.estimate", "local_ai.evaluation", "local_ai.training.finetune", "local_ai.evaluation", "local_ai.evaluation.compare", "local_ai.training.hub", "local_ai.training.calibrate"])
        (_, data), _, (_, before), (_, train), (_, after), (_, compare), (_, push), _ = outputs
        self.assertEqual(sum(item["rows"] for item in data["presets"].values()), 2000)
        self.assertEqual(data["output"], "data/processed/hf_sft")
        self.assertEqual((before["model"]["name"], after["model"]["name"], after["model"]["max_new_tokens"]), ("smoke", "smoke-colab", 256))
        self.assertEqual((train["qlora"], train["base_model"]["dtype"], train["hub"]["hub_model_id"], train["hub"]["hub_strategy"], train["hub"]["private"]), (True, "float16", HUB_REPO, "checkpoint", True))
        self.assertEqual((compare["before"], compare["after"]), (f"{before['output']}/report.json", f"{after['output']}/report.json"))
        self.assertEqual((push["status"], push["repo_id"], push["private"]), ("dry-run", HUB_REPO, True))
        self.assertEqual(push["adapter_dir"], after["model"]["adapter_path"])  # đẩy đúng adapter vừa chấm
        self.assertEqual(push["adapter_dir"], train["output_dir"] + "/adapter")  # và đúng adapter vừa train


class ColabConfigTests(unittest.TestCase):
    def test_colab_config_is_qlora_fp16_on_smoke(self):
        config = finetune.load_finetune_config(COLAB_CONFIG)
        plan = finetune.describe(config)
        self.assertEqual((plan["method"], plan["quantization"], plan["base_model"]["name"], plan["base_model"]["dtype"]), ("lora", "4bit", "smoke", "float16"))
        self.assertEqual(plan["hub"], {"push_to_hub": False})
        smoke, colab = find_model_config(PLATFORM, "smoke"), find_model_config(PLATFORM, "smoke-colab")
        self.assertEqual((colab.source, colab.dtype, colab.adapter_path), (smoke.source, "float16", f"{config.output_dir}/adapter"))

    def test_invalid_hub_and_dtype_settings_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "tên-người-dùng/tên-repo"): finetune.load_finetune_config(COLAB_CONFIG, push_to_hub=True)
        with self.assertRaisesRegex(ValueError, "dtype"): finetune.load_finetune_config(COLAB_CONFIG, dtype="float8")
        for name in ("", "huyen", "a/b/c", "../huyen", "ten ban/huyen", "nguoi-dung/"):
            with self.subTest(name), self.assertRaises(ValueError): hub.check_repo_id(name)
        self.assertEqual(hub.check_repo_id(HUB_REPO), HUB_REPO)


class HubTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(); self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_push_adapter_creates_private_repo_and_uploads(self):
        with mock.patch.dict(sys.modules, {"huggingface_hub": None}):  # dry-run không cần thư viện, không gọi mạng
            self.assertEqual(hub.push_adapter(HUB_REPO, self.root / "adapter", dry_run=True)["status"], "dry-run")
        calls = []
        with fake_hub(calls), self.assertRaisesRegex(FileNotFoundError, "adapter_config.json"): hub.push_adapter(HUB_REPO, self.root / "adapter")
        (self.root / "adapter").mkdir(); (self.root / "adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
        with fake_hub(calls): result = hub.push_adapter(HUB_REPO, self.root / "adapter")
        self.assertEqual(result["status"], "pushed")
        self.assertEqual(calls[0], ("create_repo", {"repo_id": HUB_REPO, "private": True, "exist_ok": True}))
        self.assertEqual((calls[1][0], calls[1][1]["folder_path"]), ("upload_folder", str(self.root / "adapter")))
        with fake_hub(calls), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(hub.main(["push-adapter", "--repo", HUB_REPO, "--adapter-dir", str(self.root / "adapter"), "--public"]), 0)
        self.assertFalse(json.loads(output.getvalue())["private"])
        self.assertEqual(calls[-2][1]["private"], False)

    def test_download_last_checkpoint(self):
        calls = []
        with fake_hub(calls, files={"last-checkpoint/trainer_state.json": "{}", "last-checkpoint/adapter_model.safetensors": "x"}):
            target = hub.download_last_checkpoint(HUB_REPO, self.root)
        self.assertEqual(target, self.root / "_hub" / "last-checkpoint")
        self.assertEqual(calls[0][1]["allow_patterns"], ["last-checkpoint/*"])
        self.assertEqual(hub.downloaded_checkpoint(self.root), target)

    def test_missing_or_incomplete_checkpoint_starts_from_scratch(self):
        for error in (RepositoryNotFoundError("chưa có repo"), RevisionNotFoundError("x"), EntryNotFoundError("x")):
            with self.subTest(type(error).__name__), fake_hub([], error=error): self.assertIsNone(hub.download_last_checkpoint(HUB_REPO, self.root))
        with fake_hub([], files={"last-checkpoint/adapter_model.safetensors": "x"}):  # thiếu trainer_state.json
            self.assertIsNone(hub.download_last_checkpoint(HUB_REPO, self.root))
        self.assertFalse((self.root / "_hub" / "last-checkpoint").exists())
        with fake_hub([], error=OSError("mất mạng")), self.assertRaises(OSError):  # lỗi khác thì báo ra, không lặng lẽ train lại từ đầu
            hub.download_last_checkpoint(HUB_REPO, self.root)


class FinetuneHubTests(unittest.TestCase):
    """Train với thư viện giả: SFTConfig nhận tham số đẩy Hub, chạy lại thì tải last-checkpoint để train tiếp."""

    def run_train(self, hub_files=None, local_checkpoint=False, **overrides):
        calls = []

        class Trainer:
            def __init__(self, **kwargs): calls.append(("SFTTrainer", kwargs))
            def train(self, resume_from_checkpoint): calls.append(("train", resume_from_checkpoint)); return types.SimpleNamespace(metrics={"train_loss": 1.0})
            def save_model(self, path): calls.append(("save_model", path))

        class Auto:
            def __init__(self, name, result=None): self.name, self.result = name, result
            def from_pretrained(self, source, **kwargs):
                calls.append((self.name, kwargs))
                return self.result or types.SimpleNamespace(config=types.SimpleNamespace())

        tokenizer = types.SimpleNamespace(pad_token=None, eos_token="<eos>", save_pretrained=lambda path: None)
        dataset = types.SimpleNamespace(select_columns=lambda columns: dataset)
        modules = {
            "torch": fake_module("torch", bfloat16="bf16", float16="fp16", float32="fp32", cuda=types.SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: False)),
            "transformers": fake_module("transformers", AutoTokenizer=Auto("AutoTokenizer", tokenizer), AutoModelForCausalLM=Auto("AutoModelForCausalLM"), BitsAndBytesConfig=lambda **kwargs: ("BitsAndBytesConfig", kwargs)),
            "bitsandbytes": fake_module("bitsandbytes"),
            "trl": fake_module("trl", SFTConfig=lambda **kwargs: kwargs, SFTTrainer=Trainer),
            "peft": fake_module("peft", LoraConfig=lambda **kwargs: kwargs, prepare_model_for_kbit_training=lambda model, **kwargs: model),
            "datasets": fake_module("datasets", load_dataset=lambda *args, **kwargs: dataset)}
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "sft.jsonl"; data.write_text('{"messages": []}\n', encoding="utf-8")
            output = Path(directory) / "out"
            if local_checkpoint: (output / "checkpoint-50").mkdir(parents=True)
            config = finetune.load_finetune_config(COLAB_CONFIG, dataset_path=str(data), output_dir=str(output), **overrides)
            with mock.patch.dict(sys.modules, modules), fake_hub(calls, files=hub_files), contextlib.redirect_stderr(io.StringIO()):
                result = finetune.train(config)
            relative = lambda value: str(Path(value).relative_to(output)) if isinstance(value, str) and value.startswith(str(output)) else value
            return result, [(name, relative(value)) for name, value in calls]

    def test_push_to_hub_arguments_and_fp16(self):
        result, calls = self.run_train(push_to_hub=True, hub_model_id=HUB_REPO)
        self.assertEqual(result["status"], "completed")
        args = next(value for name, value in calls if name == "SFTTrainer")["args"]
        self.assertEqual({key: args[key] for key in ("push_to_hub", "hub_model_id", "hub_strategy", "hub_private_repo", "fp16", "bf16")},
                         {"push_to_hub": True, "hub_model_id": HUB_REPO, "hub_strategy": "checkpoint", "hub_private_repo": True, "fp16": True, "bf16": False})
        load = next(value for name, value in calls if name == "AutoModelForCausalLM")
        self.assertEqual((load["torch_dtype"], load["quantization_config"][1]["load_in_4bit"]), ("fp16", True))
        self.assertIn(("train", None), calls)  # lần chạy đầu: Hub chưa có checkpoint thì train từ đầu

    def test_rerun_resumes_from_hub_checkpoint(self):
        files = {"last-checkpoint/trainer_state.json": '{"global_step": 50}'}
        _, calls = self.run_train(hub_files=files, push_to_hub=True, hub_model_id=HUB_REPO)
        self.assertEqual([name for name, _ in calls if name == "snapshot_download"], ["snapshot_download"])
        self.assertIn(("train", str(Path("_hub") / "last-checkpoint")), calls)

    def test_local_checkpoint_or_no_resume_skips_download(self):
        files = {"last-checkpoint/trainer_state.json": "{}"}
        for label, options, resume in (("checkpoint trên máy", {"local_checkpoint": True}, "checkpoint-50"), ("--no-resume", {"resume": False}, None)):
            with self.subTest(label):
                _, calls = self.run_train(hub_files=files, push_to_hub=True, hub_model_id=HUB_REPO, **options)
                self.assertNotIn("snapshot_download", [name for name, _ in calls])
                self.assertIn(("train", resume), calls)
        _, calls = self.run_train(hub_files=files)  # không bật --push-to-hub thì không gọi Hub
        self.assertEqual([name for name, _ in calls if name in ("snapshot_download", "create_repo", "upload_folder")], [])
        self.assertNotIn("push_to_hub", next(value for name, value in calls if name == "SFTTrainer")["args"])


class CompareTests(unittest.TestCase):
    @staticmethod
    def report(model, failures, groups, languages):
        passed = sum(item[0] for item in groups.values())
        return {"status": "completed", "model": model, "cases": 10, "passed": passed, "failures": [{"id": name} for name in failures],
                "groups": {name: {"passed": p, "cases": c} for name, (p, c) in groups.items()}, "languages": {name: {"passed": p, "cases": c} for name, (p, c) in languages.items()}}

    def test_table_shows_changes_and_flipped_cases(self):
        before = self.report("smoke", ["vi-01", "code-02", "code-03", "vi-04"], {"code": (3, 5), "vi": (3, 5)}, {"vi": (3, 5), "en": (3, 5)})
        after = self.report("smoke-colab", ["code-03", "vi-09", "code-07"], {"code": (3, 5), "vi": (4, 5)}, {"vi": (4, 5), "en": (3, 5)})
        table = compare_reports(before, after)
        self.assertIn("smoke (trước) → smoke-colab (sau)", table)
        self.assertIn("| Tổng | 6/10 (60%) | 7/10 (70%) | +10 điểm % |", table)
        self.assertIn("| nhóm vi | 3/5 (60%) | 4/5 (80%) | +20 điểm % |", table)
        self.assertIn("| ngôn ngữ en | 3/5 (60%) | 3/5 (60%) | +0 điểm % |", table)
        self.assertIn("Câu mới đạt: code-02, vi-01, vi-04", table)
        self.assertIn("Câu mới trượt: code-07, vi-09", table)
        with self.assertRaisesRegex(ValueError, "completed"): compare_reports({**before, "status": "skipped"}, after)

    def test_cli_reads_reports_and_reports_missing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("truoc.json", "sau.json")]
            report = self.report("smoke", [], {"code": (5, 5)}, {"en": (5, 5)})
            for path in paths: path.write_text(json.dumps({**report, "passed": 5}), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()) as output: self.assertEqual(compare_main([str(path) for path in paths]), 0)
            self.assertIn("| Tổng | 5/10 (50%) | 5/10 (50%) | +0 điểm % |", output.getvalue())
            with contextlib.redirect_stderr(io.StringIO()) as error: self.assertEqual(compare_main([str(paths[0]), str(Path(directory) / "khong-co.json")]), 2)
            self.assertIn("hãy chạy eval trước", error.getvalue())


class ReadmeTests(unittest.TestCase):
    def test_readme_has_open_in_colab_button_for_main_branch(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertTrue(f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB_LINK})" in readme, "README thiếu nút Open in Colab trỏ tới notebook ở nhánh main")
        self.assertTrue(COLAB_LINK.endswith(str(NOTEBOOK.relative_to(ROOT))))


if __name__ == "__main__":
    unittest.main()
