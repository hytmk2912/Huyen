"""Mốc M18: dùng model đã train trong agent.

Gộp LoRA adapter vào model gốc, xuất GGUF (llama.cpp) và Modelfile cho Ollama, tải adapter từ repo riêng tư, so sánh 2 lần
chạy agent (model gốc và model đã train), và notebook Colab làm cả chuỗi. Không cần mạng hay GPU: model tí hon kiến trúc
Qwen2 và Qwen3 khởi tạo ngẫu nhiên trên CPU, llama.cpp giả (script ghi file thay cho convert_hf_to_gguf.py),
huggingface_hub giả. Chạy thật với llama.cpp và Ollama: xem bằng chứng tiêu chí 2 trong TASKS.md.
"""
import contextlib
import hashlib
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
import textwrap
import types
import unittest
from pathlib import Path
from unittest import mock

from local_ai.agents import compare as agent_compare
from local_ai.config.settings import find_model_config
from local_ai.data.secrets import SECRET
from local_ai.training import export, hub

ROOT = Path(__file__).resolve().parent.parent
PLATFORM = ROOT / "configs" / "models" / "platform.json"
NOTEBOOK = ROOT / "notebooks" / "agent_trained_colab.ipynb"
LIBRARIES = ("torch", "transformers", "peft", "tokenizers")
MISSING = [name for name in LIBRARIES if importlib.util.find_spec(name) is None]
VIETNAMESE = re.compile(r"[ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]", re.I)
FAKE_CONVERTER = textwrap.dedent('''
    """convert_hf_to_gguf.py giả: kiểm tra thư mục safetensors có đủ file, rồi ghi một file "GGUF" nhỏ mô tả đầu vào."""
    import argparse, json, os, sys
    from pathlib import Path
    parser = argparse.ArgumentParser(); parser.add_argument("model"); parser.add_argument("--outfile"); parser.add_argument("--outtype")
    args = parser.parse_args()
    if os.environ.get("FAKE_CONVERTER_FAIL"):
        print("ModuleNotFoundError: No module named 'sentencepiece'", file=sys.stderr); sys.exit(1)
    folder = Path(args.model); files = sorted(path.name for path in folder.iterdir())
    for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"):
        if name not in files: print("thiếu " + name, file=sys.stderr); sys.exit(2)
    config = json.loads((folder / "config.json").read_text())
    Path(args.outfile).write_bytes(b"GGUF" + json.dumps({"model_type": config["model_type"], "outtype": args.outtype, "files": files}).encode())
''')


def helpers(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_helpers_m18", Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def fake_module(name, **attributes):
    module = types.ModuleType(name); module.__spec__ = importlib.machinery.ModuleSpec(name, None)
    for key, value in attributes.items(): setattr(module, key, value)
    return module


class RepositoryNotFoundError(Exception): pass


def fake_hub(remote: dict[str, bytes] | None, calls: list):
    """huggingface_hub giả: snapshot_download chép các file khớp allow_patterns từ `remote` (None: repo không tồn tại)."""
    def snapshot_download(repo_id, allow_patterns, local_dir):
        calls.append((repo_id, tuple(allow_patterns)))
        if remote is None: raise RepositoryNotFoundError(repo_id)
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        for name, content in remote.items():
            if name in allow_patterns: (Path(local_dir) / name).write_bytes(content)
        return local_dir
    errors = fake_module("huggingface_hub.errors", RepositoryNotFoundError=RepositoryNotFoundError)
    return mock.patch.dict(sys.modules, {"huggingface_hub": fake_module("huggingface_hub", snapshot_download=snapshot_download), "huggingface_hub.errors": errors})


@unittest.skipIf(MISSING, f"thiếu thư viện: {', '.join(MISSING)}")
class ExportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(); self.root = Path(self.directory.name)
        self.environ = mock.patch.dict(os.environ, helpers("test_m6_cpu_pipeline").OFFLINE); self.environ.start()
        self.llama = self.root / "llama.cpp"; self.llama.mkdir(); (self.llama / "convert_hf_to_gguf.py").write_text(FAKE_CONVERTER, encoding="utf-8")

    def tearDown(self):
        self.environ.stop(); self.directory.cleanup()

    def make(self, architecture: str) -> Path:
        """Model tí hon Qwen2 hoặc Qwen3 (tokenizer tí hon của M6, có chat template) + adapter LoRA khác 0; trả về file danh sách model."""
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM
        base = self.root / architecture / "base"
        helpers("test_m6_cpu_pipeline").make_tiny_model(base, ["Xin chào, đây là câu hỏi.", "Đây là câu trả lời ngắn gọn."])
        for name in ("config.json", "model.safetensors", "generation_config.json"): (base / name).unlink(missing_ok=True)  # bỏ model Llama, giữ tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base)
        common = dict(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=256, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        torch.manual_seed(0)
        model = Qwen2ForCausalLM(Qwen2Config(**common)) if architecture == "qwen2" else Qwen3ForCausalLM(Qwen3Config(head_dim=16, **common))
        model.save_pretrained(base)
        get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules="all-linear", init_lora_weights=False)).save_pretrained(self.root / architecture / "adapter")
        entry = {"source": str(base), "revision": "main", "dtype": "float32", "device_map": None, "offline": True, "kind": "text", "params_b": 0.0001, "capabilities": ["chat"]}
        models = self.root / architecture / "models.json"
        models.write_text(json.dumps({"models": [{"name": "tiny", **entry}, {"name": "tiny-colab", **entry, "adapter_path": str(self.root / architecture / "adapter")}]}), encoding="utf-8")
        return models

    def logits(self, model_dir: Path, adapter: Path | None = None):
        import torch
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
        with torch.no_grad(): return model(torch.tensor([[3, 7, 11, 19, 23]])).logits

    def run_export(self, models: Path, output: Path, *extra: str) -> dict:
        with contextlib.redirect_stdout(io.StringIO()) as printed, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(export.main(["--model", "tiny-colab", "--models", str(models), "--output", str(output), "--llama-cpp", str(self.llama), *extra]), 0)
        return json.loads(printed.getvalue())

    def test_merged_model_matches_base_plus_adapter_for_qwen2_and_qwen3(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        for architecture in ("qwen2", "qwen3"):  # họ của smoke (Qwen2.5) và light (Qwen3)
            with self.subTest(architecture):
                models = self.make(architecture); base, adapter = self.root / architecture / "base", self.root / architecture / "adapter"
                info = self.run_export(models, self.root / architecture / "da-train", "--keep-hf")
                hf = Path(info["hf_dir"])
                merged = AutoModelForCausalLM.from_pretrained(hf)
                self.assertFalse([name for name, _ in merged.named_parameters() if "lora_" in name])  # không còn lớp LoRA
                self.assertTrue(torch.allclose(self.logits(hf), self.logits(base, adapter), atol=1e-4))  # giống model gốc + adapter
                self.assertFalse(torch.allclose(self.logits(hf), self.logits(base), atol=1e-3))  # adapter thật sự làm thay đổi model
                self.assertTrue(AutoTokenizer.from_pretrained(hf).chat_template)  # tokenizer có chat template cho llama.cpp
                expected_sha = hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest()
                self.assertEqual((info["merged"], info["model_type"], info["outtype"], info["adapter"]["sha256"]["adapter_model.safetensors"]), (True, architecture, "q8_0", expected_sha))
                gguf = json.loads(Path(info["gguf"]).read_bytes()[4:])
                self.assertEqual((gguf["model_type"], gguf["outtype"]), (architecture, "q8_0"))
                saved = json.loads((self.root / architecture / "da-train" / "export_info.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["model"], "tiny-colab")

    def test_base_model_goes_through_the_same_path_without_adapter(self):
        import torch
        models = self.make("qwen3")
        trained = self.run_export(models, self.root / "da-train")
        base = self.run_export(models, self.root / "goc", "--no-adapter", "--keep-hf")
        self.assertTrue(torch.allclose(self.logits(Path(base["hf_dir"])), self.logits(self.root / "qwen3" / "base"), atol=1e-6))
        self.assertEqual((base["merged"], base["adapter"]), (False, None))
        for key in ("source", "revision", "dtype", "outtype", "model_type"):  # cùng một đường xuất: so sánh công bằng
            with self.subTest(key): self.assertEqual(base[key], trained[key])
        self.assertIsNone(trained["hf_dir"]); self.assertFalse((self.root / "da-train" / "hf").exists())  # mặc định xóa safetensors trung gian

    def test_modelfile_uses_the_ollama_template_of_the_family(self):
        for architecture, source, system in (("qwen2", "qwen2.5:0.5b", True), ("qwen3", "qwen3:4b", False)):
            with self.subTest(architecture):
                info = self.run_export(self.make(architecture), self.root / architecture / "out")
                settings = json.loads((ROOT / "configs" / "ollama" / f"{architecture}.json").read_text(encoding="utf-8"))
                text = Path(info["modelfile"]).read_text(encoding="utf-8")
                self.assertTrue(text.startswith(f"FROM {Path(info['gguf']).resolve()}\n"))
                self.assertIn(f'TEMPLATE """{settings["template"]}"""', text)
                self.assertEqual('SYSTEM """' in text, system)
                self.assertIn('PARAMETER stop "<|im_start|>"\nPARAMETER stop "<|im_end|>"', text)
                self.assertEqual(info["ollama_template_from"], source)

    def test_errors_are_reported_in_vietnamese(self):
        models = self.make("qwen2")
        with mock.patch.object(export, "save_hf", side_effect=AssertionError("không được nạp model")), self.assertRaises(FileNotFoundError) as caught:
            export.export(export.export_model_config("tiny-colab", models), self.root / "out", self.root / "khong-co-llama")
        self.assertIn("git clone --depth 1 --branch b11205", str(caught.exception))  # thiếu llama.cpp: báo trước khi nạp model
        with mock.patch.dict(os.environ, {"FAKE_CONVERTER_FAIL": "1"}), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(RuntimeError) as caught:
            export.export(export.export_model_config("tiny-colab", models), self.root / "out", self.llama)
        self.assertIn("Chuyển sang GGUF thất bại", str(caught.exception)); self.assertIn("python -m pip install sentencepiece", str(caught.exception))
        missing = json.loads(models.read_text(encoding="utf-8")); missing["models"][1]["adapter_path"] = str(self.root / "khong-co")
        models.write_text(json.dumps(missing), encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaisesRegex(FileNotFoundError, "pull-adapter"):
            export.export(export.export_model_config("tiny-colab", models), self.root / "out2", self.llama)


class ExportPlanTests(unittest.TestCase):
    def dry_run(self, *extra: str) -> dict:
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            self.assertEqual(export.main(["--output", ".runs/gguf/x", "--llama-cpp", "/khong/co", "--dry-run", *extra]), 0)
        return json.loads(printed.getvalue())

    def test_dry_run_uses_pinned_revision_and_same_path_for_both_models(self):
        for model in ("smoke", "light"):
            with self.subTest(model):
                trained, base = self.dry_run("--model", f"{model}-colab"), self.dry_run("--model", f"{model}-colab", "--no-adapter")
                config = find_model_config(PLATFORM, f"{model}-colab")
                self.assertEqual((trained["merged"], trained["adapter_path"], base["merged"], base["adapter_path"]), (True, config.adapter_path, False, None))
                for key in ("source", "revision", "dtype", "outtype"):
                    with self.subTest(key): self.assertEqual(trained[key], base[key])
                self.assertRegex(trained["revision"], r"^[0-9a-f]{40}$")
                self.assertEqual((trained["dtype"], trained["outtype"], trained["converter_found"]), ("float16", "q8_0", False))

    def test_only_local_text_models_can_be_exported(self):
        for name, message in (("ollama-colab", "chạy qua server"), ("primary", "ảnh \\+ chữ"), ("primary-qlora", "ảnh \\+ chữ")):
            with self.subTest(name), self.assertRaisesRegex(ValueError, message): export.export_model_config(name, PLATFORM)
        self.assertIsNone(export.export_model_config("smoke-colab", PLATFORM).quantization)  # gộp trên model không nén

    def test_ollama_templates_are_verbatim_copies(self):
        for name, source in (("qwen2", "qwen2.5:0.5b"), ("qwen3", "qwen3:4b")):
            with self.subTest(name):
                settings = export.ollama_settings(name)
                self.assertEqual(settings["ollama_source"], source)
                self.assertEqual(hashlib.sha256(settings["template"].encode()).hexdigest(), settings["template_sha256"])
                self.assertEqual([value for key, value in settings["parameters"] if key == "stop"], ["<|im_start|>", "<|im_end|>"])
        qwen3 = export.ollama_settings("qwen3")  # lớp params của qwen3:4b: JSON kiểu Go, thứ tự khóa theo tên, có xuống dòng cuối
        params = {}
        for key, value in qwen3["parameters"]: params.setdefault(key, []).append(value)
        raw = json.dumps({key: value if len(value) > 1 else value[0] for key, value in params.items()}, separators=(",", ":"), sort_keys=True)
        self.assertIn(hashlib.sha256((raw.replace("<", "\\u003c").replace(">", "\\u003e") + "\n").encode()).hexdigest(), qwen3["_comment"])
        with self.assertRaisesRegex(ValueError, "model_type 'llama'"): export.ollama_settings("llama")
        edited = {**qwen3, "template": qwen3["template"] + " "}
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(export, "OLLAMA_CONFIGS", Path(directory)), mock.patch.object(export, "ROOT", Path(directory)):
            (Path(directory) / "qwen3.json").write_text(json.dumps(edited), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "đừng sửa tay"): export.ollama_settings("qwen3")
        with self.assertRaisesRegex(ValueError, "ba dấu nháy"): export.modelfile(Path("model.gguf"), {"template": 'a """ b'})


class PullAdapterTests(unittest.TestCase):
    def test_downloads_only_adapter_files(self):
        calls = []
        remote = {"adapter_config.json": b"{}", "adapter_model.safetensors": b"trong-so", "last-checkpoint/optimizer.pt": b"x", "measurements.json": b"{}"}
        with tempfile.TemporaryDirectory() as directory, fake_hub(remote, calls):
            result = hub.pull_adapter("nguoi-dung/huyen-light-qlora", Path(directory) / "adapter")
            files = sorted(path.name for path in (Path(directory) / "adapter").iterdir())
        self.assertEqual((result["status"], result["files"], files), ("pulled", ["adapter_config.json", "adapter_model.safetensors"], ["adapter_config.json", "adapter_model.safetensors"]))
        self.assertEqual(calls, [("nguoi-dung/huyen-light-qlora", hub.ADAPTER_FILES)])

    def test_missing_repo_or_adapter_is_explained(self):
        with tempfile.TemporaryDirectory() as directory:
            for remote in (None, {"measurements.json": b"{}"}):
                with self.subTest(remote), fake_hub(remote, []), self.assertRaisesRegex(FileNotFoundError, "train_colab tới Bước 11"):
                    hub.pull_adapter("nguoi-dung/huyen-light-qlora", Path(directory) / "adapter")
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            self.assertEqual(hub.main(["pull-adapter", "--repo", "nguoi-dung/x", "--adapter-dir", ".runs/colab_smoke/adapter", "--dry-run"]), 0)
        self.assertEqual(json.loads(printed.getvalue())["status"], "dry-run")
        with self.assertRaisesRegex(ValueError, "tên-người-dùng/tên-repo"): hub.pull_adapter("khong-hop-le", ".runs/x", dry_run=True)


def report(model: str, results: dict[str, tuple[bool, str | None]], minutes: float) -> dict:
    tasks = [{"id": identifier, "success": success, "reason": reason, "tools_used": ["terminal"], "duration_s": 60.0} for identifier, (success, reason) in results.items()]
    passed = sum(success for success, _ in results.values())
    return {"status": "completed", "model": model, "tasks": tasks, "passed": passed, "total": len(tasks), "success_rate": round(passed / len(tasks), 4), "duration_s": minutes * 60}


class AgentCompareTests(unittest.TestCase):
    BASE = report("ollama-light-goc", {"tinh-bieu-thuc": (True, None), "tong-cot-csv": (False, "câu trả lời thiếu 87"), "doc-ghi-chu": (True, None), "chi-o-lan-1": (True, None)}, 11.1)
    TRAINED = report("ollama-light-da-train", {"tinh-bieu-thuc": (True, None), "tong-cot-csv": (True, None), "doc-ghi-chu": (False, "agent không hoàn thành"), "chi-o-lan-2": (False, "x")}, 9.5)

    def test_table_rates_and_changes(self):
        result = agent_compare.compare_reports(self.BASE, self.TRAINED, ("model gốc", "model đã train"))
        self.assertEqual((result["newly_passed"], result["newly_failed"], result["only_in_one"]), (["tong-cot-csv"], ["doc-ghi-chu"], ["chi-o-lan-1", "chi-o-lan-2"]))
        text = agent_compare.format_comparison(result)
        for phrase in ("| Nhiệm vụ | model gốc | model đã train |", "| tong-cot-csv | không đạt (câu trả lời thiếu 87) | ĐẠT |", "| chi-o-lan-2 | — | không đạt (x) |",
                       "| **Tỉ lệ thành công** | 3/4 (75%) | 2/4 (50%) |", "| Thời gian | 11,1 phút | 9,5 phút |", "Mới đạt ở model đã train: tong-cot-csv.", "Mới trượt ở model đã train: doc-ghi-chu."):
            with self.subTest(phrase): self.assertIn(phrase, text)

    def test_cli_reads_task_reports_and_writes_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in (("goc.json", self.BASE), ("da_train.json", self.TRAINED), ("loi.json", {"status": "error", "error": "x"})):
                (root / name).write_text(json.dumps(value), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                code = agent_compare.main([str(root / "goc.json"), str(root / "da_train.json"), "--labels", "model gốc", "model đã train", "--output", str(root / "so_sanh.json")])
            saved = json.loads((root / "so_sanh.json").read_text(encoding="utf-8"))
            with self.assertRaisesRegex(ValueError, "chưa|không phải báo cáo agent"): agent_compare.main([str(root / "loi.json"), str(root / "goc.json")])
        self.assertEqual(code, 0); self.assertIn("So sánh agent: model gốc (ollama-light-goc)", printed.getvalue())
        self.assertEqual(saved["models"], ["ollama-light-goc", "ollama-light-da-train"])

    def test_scripted_task_reports_can_be_compared(self):
        tasks = helpers("test_m13_agent_colab")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("a", "b"):
                tasks.run_cli(["--scripted", "--workspace", str(root / name), "--output", str(root / f"{name}.json")])
            result = agent_compare.compare_reports(agent_compare.load_report(root / "a.json"), agent_compare.load_report(root / "b.json"))
        self.assertEqual((result["summary"][0]["passed"], result["newly_passed"], result["newly_failed"]), (6, [], []))


class PlatformEntriesTests(unittest.TestCase):
    def test_ollama_entries_for_base_and_trained_models(self):
        for model, params in (("smoke", 0.49), ("light", 4.02)):
            for variant in ("goc", "da-train"):
                with self.subTest(f"{model}-{variant}"):
                    config = find_model_config(PLATFORM, f"ollama-{model}-{variant}")
                    self.assertEqual((config.backend, config.base_url, config.source, config.params_b), ("openai_compatible", "http://localhost:11434/v1", f"huyen-{model}-{variant}", params))
                    self.assertIn("reasoning", {capability.value for capability in config.capabilities})  # agent cần khả năng reasoning


def notebook_cells() -> dict[str, str]:
    return {cell["id"]: "".join(cell["source"]) for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"] if cell["cell_type"] == "code"}


def notebook_commands(model: str) -> list[list[str]]:
    variables = {"MODEL": model, "HUB_REPO": f"nguoi-dung/huyen-{model}-qlora"}
    commands = []
    for source in notebook_cells().values():
        for command in re.findall(r'run\(f?"(python -m local_ai[^"]*)"', source):
            for name, extra in ((("goc", "--no-adapter"), ("da-train", "")) if "{extra}" in command else ((None, None),)):  # vòng lặp của Bước 6
                values = {**variables, "name": name, "extra": extra or ""}
                commands.append(shlex.split(re.sub(r"\{(\w+)\}", lambda match: values[match.group(1)], command)))
    return commands


class NotebookTests(unittest.TestCase):
    def test_valid_matches_builder_and_has_no_outputs(self):
        import nbformat
        nbformat.validate(nbformat.read(str(NOTEBOOK), as_version=4))
        spec = importlib.util.spec_from_file_location("build_notebooks_m18", ROOT / "notebooks" / "build.py")
        build = importlib.util.module_from_spec(spec); spec.loader.exec_module(build)
        self.assertEqual(NOTEBOOK.read_text(encoding="utf-8"), build.render(build.agent_trained_colab), "file .ipynb khác notebooks/build.py: hãy chạy python notebooks/build.py")
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                with self.subTest(cell["id"]): self.assertEqual((cell["outputs"], cell["execution_count"]), ([], None))
        cells = notebook_cells()
        self.assertEqual(list(cells), ["buoc-1-chon-model", "buoc-2-gpu", "buoc-3-cai-dat", "buoc-4-token", "buoc-5-tai-adapter", "buoc-6-xuat-gguf", "buoc-7-chay-ollama",
                                       "buoc-8-tao-model", "buoc-9-agent-goc", "buoc-10-agent-da-train", "buoc-11-so-sanh"])
        for number, (cell_id, source) in enumerate(cells.items(), start=1):
            with self.subTest(cell_id):
                first = source.splitlines()[0]
                self.assertTrue(first.startswith(f"# Bước {number}:"), first); self.assertRegex(first, VIETNAMESE)
                for step in re.findall(r'run\((?:f?"[^"]*"|\[[^\]]*\]), f?"([^"]+)"', source): self.assertTrue(step.startswith(f"Bước {number} ("), step)
                escapes = [line.strip() for line in source.splitlines() if line.strip().startswith("!")]
                self.assertTrue(all(line.startswith("!git clone") for line in escapes), escapes)  # chỉ lần tải code đầu tiên

    def test_pinned_versions_token_and_model_names(self):
        spec = importlib.util.spec_from_file_location("build_notebooks_m18b", ROOT / "notebooks" / "build.py")
        build = importlib.util.module_from_spec(spec); spec.loader.exec_module(build)
        cells = notebook_cells()
        packages = re.search(r'"pip install -q ([^"]+)"', cells["buoc-3-cai-dat"]).group(1).split()
        self.assertTrue(packages and all(re.fullmatch(r"[a-z][a-z0-9_-]*==[0-9][0-9.]*", name) for name in packages), packages)
        self.assertNotIn("torch", [name.split("==")[0] for name in packages]); self.assertIn("sentencepiece", [name.split("==")[0] for name in packages])
        self.assertIn(f'"--branch", "{build.LLAMA_CPP_TAG}"', cells["buoc-3-cai-dat"]); self.assertIn(build.LLAMA_CPP_TAG, export.LLAMA_CPP_HINT)
        self.assertIn(f'"OLLAMA_VERSION": "{build.OLLAMA_VERSION}"', cells["buoc-3-cai-dat"])
        self.assertIn('userdata.get("HF_TOKEN")', cells["buoc-4-token"])
        text = NOTEBOOK.read_text(encoding="utf-8")
        self.assertIsNone(SECRET.search(text))
        self.assertFalse([line for source in cells.values() for line in source.splitlines() if "print(" in line and "HF_TOKEN" in line])
        for variant in ("goc", "da-train"):  # tên model trong Ollama khớp mục ollama-<model>-<variant> trong danh sách model
            with self.subTest(variant):
                self.assertIn(f"ollama create huyen-{{MODEL}}-{variant} -f .runs/gguf/{{MODEL}}-{variant}/Modelfile", cells["buoc-8-tao-model"])
                self.assertIn(f"--model ollama-{{MODEL}}-{variant}", cells[f"buoc-{9 if variant == 'goc' else 10}-agent-{variant}"])

    def test_commands_run_with_dry_run(self):
        environ = {**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONPATH": str(ROOT)}
        for model in ("smoke", "light"):
            commands = notebook_commands(model)
            self.assertEqual([command[2] for command in commands], ["local_ai.training.hub", "local_ai.training.export", "local_ai.training.export", "local_ai.agents.tasks", "local_ai.agents.tasks", "local_ai.agents.compare"])
            outputs = []
            for command in commands:
                with self.subTest(f"{model}: {' '.join(command)}"):
                    result = subprocess.run([sys.executable, *command[1:], "--dry-run"], cwd=ROOT, env=environ, capture_output=True, text=True, timeout=120)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    outputs.append(json.loads(result.stdout))
            pull, base, trained, agent_base, agent_trained, compare = outputs
            self.assertEqual(pull["adapter_dir"], trained["adapter_path"])  # tải adapter vào đúng chỗ lệnh xuất đọc
            self.assertEqual((base["merged"], trained["merged"]), (False, True))
            self.assertEqual((agent_base["model"]["source"], agent_trained["model"]["source"]), (f"huyen-{model}-goc", f"huyen-{model}-da-train"))
            self.assertEqual(compare["labels"], ["model gốc", "model đã train"])


if __name__ == "__main__":
    unittest.main()
