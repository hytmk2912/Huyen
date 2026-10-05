"""Hardening bảo mật (5/10, chủ repo yêu cầu, không phải mốc): chống lộ secret và chống tấn công supply-chain.

1. Secret chỉ đọc qua `local_ai/config/secrets.py`: thiếu biến bắt buộc thì dừng với lỗi tiếng Việt, không có giá trị mặc định,
   thông báo lỗi không chứa giá trị secret; HF_TOKEN không bị ghi vào dữ liệu đầu ra hay kế hoạch train.
2. Nạp model an toàn: `trust_remote_code=False`, chỉ safetensors (`use_safetensors=True`); repo hay adapter chỉ có pickle (.bin/.pkl)
   thì từ chối, trừ khi chạy với cờ `--allow-pickle-weights`. Không nơi nào gọi pickle.load hay torch.load không có weights_only=True.
3. Ghim revision: mọi model Hugging Face trong danh sách model và mọi dataset preset dùng mã commit 40 ký tự; tải dataset thật mà chưa ghim thì từ chối.
4. Chặn path traversal khi ghi dữ liệu tải về và khi đọc preset theo tên.

Test dùng loader và module giả, không cần mạng hay GPU.
"""
import ast
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from local_ai.config import secrets
from local_ai.config.settings import load_model_configs
from local_ai.data import hub as data_hub
from local_ai.models import adapters

ROOT = Path(__file__).resolve().parents[1]
PLATFORM = ROOT / "configs" / "models" / "platform.json"
SAFETENSORS = json.loads((ROOT / "tests" / "fixtures" / "hf_api" / "safetensors_2026-10-05.json").read_text(encoding="utf-8"))["models"]
FAKE_TOKEN = "hf_" + "GiaDeThu" * 4  # token giả, ghép lúc chạy để secret-scan không báo chính file test
COMMIT = "0" * 40


def python_files():
    return sorted((ROOT / "local_ai").rglob("*.py"))


def calls(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call): yield node


def call_name(node):
    func = node.func
    if isinstance(func, ast.Attribute):
        base = func.value.id if isinstance(func.value, ast.Name) else (func.value.attr if isinstance(func.value, ast.Attribute) else "")
        return f"{base}.{func.attr}"
    return func.id if isinstance(func, ast.Name) else ""


class SecretModuleTests(unittest.TestCase):
    def test_missing_required_secret_exits_with_clear_message(self):
        with self.assertRaises(SystemExit) as caught: secrets.require_secret("HF_TOKEN", {})
        self.assertIsInstance(caught.exception, secrets.MissingSecretError)
        self.assertIn("HF_TOKEN", str(caught.exception.code)); self.assertIn("Thiếu biến môi trường", str(caught.exception.code))
        for environ in ({"HF_TOKEN": ""}, {"HF_TOKEN": "   "}):
            with self.assertRaises(secrets.MissingSecretError): secrets.require_secret("HF_TOKEN", environ)

    def test_short_secret_is_rejected_without_echoing_its_value(self):
        with self.assertRaises(secrets.MissingSecretError) as caught: secrets.require_secret("LOCAL_AI_GATEWAY_TOKEN", {"LOCAL_AI_GATEWAY_TOKEN": "bimat-ngan"}, min_length=16)
        self.assertNotIn("bimat-ngan", str(caught.exception.code)); self.assertIn("16", str(caught.exception.code))

    def test_no_default_value_fallback(self):
        self.assertIsNone(secrets.optional_secret("HF_TOKEN", {}))
        self.assertEqual(secrets.require_secret("HF_TOKEN", {"HF_TOKEN": FAKE_TOKEN}), FAKE_TOKEN)
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(secrets.hf_token()); self.assertIsNone(data_hub.hf_token())
            with self.assertRaises(secrets.MissingSecretError): secrets.hf_token(required=True)

    def test_redact_hides_secret_values(self):
        text = f"lỗi khi gọi Hub với token {FAKE_TOKEN}"
        self.assertEqual(secrets.redact(text, {"HF_TOKEN": FAKE_TOKEN}), "lỗi khi gọi Hub với token ***")

    def test_secrets_are_read_only_in_the_config_module(self):
        """Chỉ local_ai/config/secrets.py đọc biến môi trường có tên trong KNOWN_SECRETS (code khác gọi qua module này)."""
        offenders = []
        for path in python_files():
            if path.name == "secrets.py" and path.parent.name == "config": continue
            for node in calls(ast.parse(path.read_text(encoding="utf-8"))):
                if call_name(node) in ("environ.get", "os.getenv", "getenv") and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value in secrets.KNOWN_SECRETS:
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
            source = path.read_text(encoding="utf-8")
            for name in secrets.KNOWN_SECRETS:
                if f'environ["{name}"]' in source: offenders.append(f"{path.relative_to(ROOT)}: environ[\"{name}\"]")
        self.assertEqual(offenders, [])

    def test_no_secret_is_printed_or_logged(self):
        """Không lệnh print/log nào trong local_ai nhận trực tiếp biến chứa token hay khóa."""
        offenders = []
        for path in python_files():
            for node in calls(ast.parse(path.read_text(encoding="utf-8"))):
                if call_name(node).split(".")[-1] not in ("print", "debug", "info", "warning", "error", "exception", "write"): continue
                for argument in ast.walk(node):
                    if isinstance(argument, ast.Name) and argument.id in ("token", "hf_token", "key", "api_key", "secret", "password"):
                        offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
        self.assertEqual(offenders, [])

    def test_env_example_lists_every_known_secret_with_empty_values(self):
        text = (ROOT / ".env.example").read_text(encoding="utf-8")
        for name in secrets.KNOWN_SECRETS: self.assertRegex(text, rf"(?m)^{name}=$")
        for line in text.splitlines():
            if line and not line.startswith("#"): self.assertRegex(line, r"^[A-Z0-9_]+=$", "giá trị trong .env.example phải để trống")

    def test_gitignore_blocks_secret_and_artifact_files(self):
        ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        for pattern in (".env", ".env.*", "*.key", "*.pem", "keystore/", "wallet*.json", "*.pkl", "checkpoints/", "outputs/", "__pycache__/", ".venv/", "!.env.example"):
            self.assertIn(pattern, ignored)
        result = subprocess.run(["git", "check-ignore", "-q", ".env.example"], cwd=ROOT)
        self.assertEqual(result.returncode, 1, ".env.example phải được commit")


class HfTokenNotPersistedTests(unittest.TestCase):
    def test_token_is_not_written_to_dataset_outputs(self):
        from tests.test_m4_presets import fixture_loader
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"HF_TOKEN": FAKE_TOKEN}):
            seen = []
            config = data_hub.load_preset("vietnamese"); config["hf_dataset"]["token"] = FAKE_TOKEN  # kể cả khi ai đó ghi token vào cấu hình
            data_hub.prepare_hf_sft(config, directory, fixture_loader(seen))
            self.assertEqual(seen[0]["token"], FAKE_TOKEN)  # token chỉ được truyền cho thư viện tải dữ liệu
            for path in Path(directory).rglob("*"):
                if path.is_file(): self.assertNotIn(FAKE_TOKEN, path.read_text(encoding="utf-8", errors="replace"), path.name)

    def test_token_is_not_in_training_plan_or_trainer_arguments(self):
        from local_ai.training.finetune import describe, hub_arguments, load_finetune_config
        with mock.patch.dict(os.environ, {"HF_TOKEN": FAKE_TOKEN}):
            config = load_finetune_config("configs/training/colab_smoke.json", push_to_hub=True, hub_model_id="nguoi-dung/thu")
            text = json.dumps([describe(config), hub_arguments(config)], ensure_ascii=False, default=str)
        self.assertNotIn(FAKE_TOKEN, text); self.assertNotIn("hub_token", text)

    def test_gateway_error_does_not_echo_token(self):
        from local_ai.runtime.gateway import GatewayConfig, GatewayRefused, create_server
        with self.assertRaises(GatewayRefused) as caught: create_server(GatewayConfig(enabled=True, port=0), {"LOCAL_AI_GATEWAY_TOKEN": "bimat-ngan"})
        self.assertNotIn("bimat-ngan", str(caught.exception))

    def test_local_server_key_is_required_once_declared(self):
        from local_ai.contracts import Message
        from local_ai.models.adapters import ModelConfig
        from local_ai.models.openai_compatible import OpenAICompatibleAdapter
        config = ModelConfig(name="srv", source="m", backend="openai_compatible", base_url="http://127.0.0.1:9/v1", api_key_env="TEST_KHOA_KHONG_CO")
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaises(secrets.MissingSecretError) as caught:
            OpenAICompatibleAdapter(config).generate([Message("user", "hi")])
        self.assertIn("TEST_KHOA_KHONG_CO", str(caught.exception.code))


class SafeModelLoadingTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(adapters.set_allow_pickle_weights, False)

    def loader(self, error=None):
        seen = []
        def from_pretrained(source, **kwargs):
            seen.append(kwargs)
            if error and kwargs.get("use_safetensors"): raise OSError(error)
            return ("model", source)
        return types.SimpleNamespace(from_pretrained=from_pretrained), seen

    def test_models_load_safetensors_without_remote_code(self):
        loader, seen = self.loader()
        self.assertEqual(adapters.from_pretrained_safely(loader, "org/m", revision=COMMIT), ("model", "org/m"))
        self.assertEqual((seen[0]["trust_remote_code"], seen[0]["use_safetensors"]), (False, True))
        self.assertEqual(adapters.safe_tokenizer_kwargs("org/m", COMMIT), {"trust_remote_code": False})

    def test_pickle_only_repo_needs_explicit_flag(self):
        loader, seen = self.loader("org/m does not appear to have a file named model.safetensors")
        with self.assertRaisesRegex(adapters.UnsafeModelError, "--allow-pickle-weights"): adapters.from_pretrained_safely(loader, "org/m", revision=COMMIT)
        with contextlib.redirect_stderr(io.StringIO()) as warning: adapters.set_allow_pickle_weights(True)
        self.assertIn("CẢNH BÁO", warning.getvalue())
        self.assertEqual(adapters.from_pretrained_safely(loader, "org/m", revision=COMMIT), ("model", "org/m"))
        self.assertEqual((seen[-1]["use_safetensors"], seen[-1]["trust_remote_code"]), (False, False))

    def test_other_load_errors_are_not_hidden(self):
        loader, _ = self.loader("mất mạng")
        with self.assertRaisesRegex(OSError, "mất mạng"): adapters.from_pretrained_safely(loader, "org/m", revision=COMMIT)

    def test_pickle_only_adapter_needs_explicit_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "adapter_model.bin").write_bytes(b"pickle")
            with self.assertRaisesRegex(adapters.UnsafeModelError, "--allow-pickle-weights"): adapters.check_adapter_files(directory)
            with contextlib.redirect_stderr(io.StringIO()): adapters.set_allow_pickle_weights(True)
            adapters.check_adapter_files(directory)
            adapters.set_allow_pickle_weights(False)
            (Path(directory) / "adapter_model.safetensors").write_bytes(b"x")
            adapters.check_adapter_files(directory)

    def test_commands_that_load_models_offer_the_flag(self):
        from local_ai.evaluation.__main__ import main as evaluation_main
        from local_ai.training.export import main as export_main
        from local_ai.training.finetune import main as finetune_main
        for main in (evaluation_main, export_main, finetune_main):
            with self.subTest(main.__module__), contextlib.redirect_stdout(io.StringIO()) as output, self.assertRaises(SystemExit): main(["--help"])
            self.assertIn("--allow-pickle-weights", output.getvalue())

    def test_every_from_pretrained_call_is_safe(self):
        """Model nạp qua from_pretrained_safely; tokenizer/processor truyền safe_tokenizer_kwargs; adapter PeftModel kiểm tra file trước."""
        offenders = []
        for path in python_files():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in calls(tree):
                for keyword in node.keywords:
                    if keyword.arg == "trust_remote_code" and not (isinstance(keyword.value, ast.Constant) and keyword.value.value is False): offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} trust_remote_code")
                if not (isinstance(node.func, ast.Attribute) and node.func.attr == "from_pretrained"): continue
                name = call_name(node)
                if name == "loader.from_pretrained" and path.name == "adapters.py": continue  # bên trong from_pretrained_safely
                if name == "PeftModel.from_pretrained": continue  # đi sau check_adapter_files (xem test dưới)
                if not any(isinstance(keyword.value, ast.Call) and call_name(keyword.value) == "safe_tokenizer_kwargs" for keyword in node.keywords if keyword.arg is None):
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} {name}")
        self.assertEqual(offenders, [])
        for path in python_files():
            source = path.read_text(encoding="utf-8")
            if "PeftModel.from_pretrained(" in source:
                with self.subTest(path.name): self.assertEqual(source.count("PeftModel.from_pretrained("), source.count("check_adapter_files(config.adapter_path)"))

    def test_no_pickle_or_unsafe_torch_load(self):
        offenders = []
        for path in python_files():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)) and any(alias.name in ("pickle", "dill", "cloudpickle", "joblib") for alias in node.names): offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} import")
                if isinstance(node, ast.ImportFrom) and node.module in ("pickle", "dill", "cloudpickle", "joblib"): offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} import")
            for node in calls(tree):
                name = call_name(node)
                if name in ("pickle.load", "pickle.loads", "pickle.Unpickler"): offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} {name}")
                if name == "torch.load" and not any(keyword.arg == "weights_only" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True for keyword in node.keywords):
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} torch.load")
        self.assertEqual(offenders, [])


class PinnedRevisionTests(unittest.TestCase):
    def test_platform_models_are_pinned_and_have_safetensors(self):
        configs = load_model_configs(PLATFORM)  # tự kiểm tra ghim revision khi đọc
        hub_models = [config for config in configs if config.backend == "transformers" and adapters.HF_REPO_ID.match(config.source)]
        self.assertGreaterEqual(len(hub_models), 5)
        for config in hub_models:
            with self.subTest(config.name):
                self.assertRegex(config.revision, r"^[0-9a-f]{40}$")
                if config.tokenizer_revision: self.assertRegex(config.tokenizer_revision, r"^[0-9a-f]{40}$")
                self.assertTrue(SAFETENSORS[config.source]["safetensors"], "model chỉ có pickle: phải dùng --allow-pickle-weights")

    def test_unpinned_hub_model_in_list_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.json"
            for entry in ({"name": "x", "source": "org/m"}, {"name": "x", "source": "org/m", "revision": "main"}, {"name": "x", "source": "org/m", "revision": COMMIT, "tokenizer_source": "org/t"}):
                path.write_text(json.dumps({"models": [entry]}), encoding="utf-8")
                with self.subTest(entry), self.assertRaisesRegex(adapters.UnsafeModelError, "chưa ghim revision"): load_model_configs(path)
            path.write_text(json.dumps({"models": [{"name": "x", "source": directory}, {"name": "y", "source": "scripted"}, {"name": "z", "source": "org/m", "revision": COMMIT}]}), encoding="utf-8")
            self.assertEqual(len(load_model_configs(path)), 3)  # thư mục trên máy, tên không phải repo Hub, hoặc đã ghim

    def test_dataset_presets_are_pinned(self):
        for item in data_hub.list_presets():
            with self.subTest(item["name"]): self.assertRegex(item["revision"], r"^[0-9a-f]{40}$")

    def test_real_download_requires_pinned_revision(self):
        spec = {"name": "org/ds", "revision": "main", "license": {"name": "CC0-1.0"}, "mapping": {"input": "q", "expected_output": "a"}}
        with mock.patch.dict(sys.modules, {"datasets": None}), self.assertRaisesRegex(ValueError, "mã commit 40 ký tự"): data_hub.load_hf_rows(spec)
        with mock.patch.dict(sys.modules, {"datasets": types.SimpleNamespace(load_dataset=lambda *args, **kwargs: iter([{"q": "1", "a": "2"}]))}):
            self.assertEqual(len(data_hub.load_hf_rows({**spec, "revision": COMMIT})), 1)


class PathTraversalTests(unittest.TestCase):
    def test_output_must_stay_in_allowed_data_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(data_hub.safe_output_dir(Path(directory) / "ok"), Path(directory) / "ok")
        self.assertEqual(data_hub.safe_output_dir("data/processed/hf_sft"), Path("data/processed/hf_sft"))
        for bad in ("data/../../.ssh", "/etc/local-ai", str(Path.home() / ".ssh" / "x"), "../ngoai-repo"):
            with self.subTest(bad), self.assertRaises(data_hub.UnsafePathError): data_hub.safe_output_dir(bad)

    def test_symlink_out_of_allowed_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            link = Path(directory) / "lien-ket"; link.symlink_to("/etc")
            with self.assertRaises(data_hub.UnsafePathError): data_hub.safe_output_dir(link / "x")

    def test_extra_directory_from_environment(self):
        target = str(Path.home() / "du-lieu-rieng")
        with mock.patch.dict(os.environ, {data_hub.DATA_DIR_ENV: target}): self.assertEqual(str(data_hub.safe_output_dir(target + "/a")), target + "/a")

    def test_downloads_refuse_unsafe_output(self):
        from tests.test_m4_presets import fixture_loader
        with self.assertRaises(data_hub.UnsafePathError): data_hub.prepare_hf_sft(data_hub.load_preset("vietnamese"), "/etc/local-ai", fixture_loader([]))
        with self.assertRaises(data_hub.UnsafePathError): data_hub.prepare_hf_mix({"code": 1.0}, "data/../../ngoai", fixture_loader([]), total=2)
        result = subprocess.run([sys.executable, "-m", "local_ai.data", "hf-sft", "--preset", "code", "--total", "4", "--output", "/etc/local-ai", "--dry-run"], cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0); self.assertIn("Không ghi dữ liệu ra", result.stderr)

    def test_preset_name_cannot_escape_preset_directory(self):
        for name in ("../models/platform", "/etc/passwd", "code/../code", ""):
            with self.subTest(name), self.assertRaisesRegex(LookupError, "Tên preset"): data_hub.load_preset(name)


class RepositorySecurityFilesTests(unittest.TestCase):
    def test_workflows_have_read_only_permissions_and_pinned_actions(self):
        import re
        workflows = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
        self.assertTrue(workflows)
        for path in workflows:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path.name):
                self.assertRegex(text, r"(?m)^permissions:\n  contents: read$")
                self.assertNotIn("pull_request_target", text)
                for line in re.findall(r"(?m)^\s*(?:-\s*)?uses:\s*(\S+)", text): self.assertRegex(line, r"@[0-9a-f]{40}$")
                for run in re.findall(r"(?m)^\s*(?:-\s*)?run:.*$", text): self.assertNotIn("${{ github.event", run)

    def test_pre_commit_and_dependabot_are_configured(self):
        pre_commit = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
        for hook in ("gitleaks", "detect-private-key", "check-added-large-files"): self.assertIn(f"id: {hook}", pre_commit)
        dependabot = (ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")
        for ecosystem in ("pip", "github-actions"): self.assertIn(f'package-ecosystem: "{ecosystem}"', dependabot)
        self.assertEqual(dependabot.count('interval: "weekly"'), 2)

    def test_requirements_are_pinned_exactly(self):
        for name in ("requirements.txt", "requirements-dev.txt"):
            for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
                line = line.split("#", 1)[0].strip()
                if line and not line.startswith("-"): self.assertRegex(line, r"^[A-Za-z0-9_.\[\]-]+==[0-9][\w.+]*$", f"{name}: {line}")

    def test_security_doc_exists_in_vietnamese(self):
        text = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
        for phrase in ("HF_TOKEN", "LOCAL_AI_GATEWAY_TOKEN", "2FA", "passkey", "ruleset", "Dependabot", "force push", "--allow-pickle-weights", "pip-audit"):
            self.assertIn(phrase, text)


if __name__ == "__main__":
    unittest.main()
