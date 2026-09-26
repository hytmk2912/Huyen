"""Mốc M2: nạp model đúng loại (text/multimodal), nén 4bit/8bit, QLoRA, tự chuyển fp16, ước tính VRAM.

Mọi test dùng module torch/transformers/trl/peft/datasets giả: không tải model, không cần mạng hay GPU.
"""
import contextlib
import importlib.machinery
import io
import os
import subprocess
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from local_ai.config.settings import find_model_config, load_model_configs
from local_ai.contracts import Message
from local_ai.models.adapters import HuggingFaceModelAdapter, ModelConfig, resolve_torch_dtype

ROOT = Path(__file__).resolve().parent.parent
PLATFORM = ROOT / "configs" / "models" / "platform.json"


def fake_module(name, **attributes):
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, None)
    for key, value in attributes.items(): setattr(module, key, value)
    return module


def fake_torch(cuda=False, bf16=True):
    return fake_module("torch", bfloat16="torch.bfloat16", float16="torch.float16", float32="torch.float32", cuda=types.SimpleNamespace(is_available=lambda: cuda, is_bf16_supported=lambda: bf16))


class FakeTokenizer:
    pad_token, eos_token = None, "<eos>"
    def save_pretrained(self, path): pass


class FakeAuto:
    """Giả một lớp Auto* của transformers: ghi lại lời gọi from_pretrained."""
    def __init__(self, name, calls, result=None): self.name, self.calls, self.result = name, calls, result
    def from_pretrained(self, source, **kwargs):
        self.calls.append((self.name, source, kwargs))
        return self.result if self.result is not None else types.SimpleNamespace(loaded_by=self.name, config=types.SimpleNamespace(), device="cpu")


def fake_transformers(calls, classes=("AutoModelForMultimodalLM", "AutoModelForImageTextToText")):
    attributes = {name: FakeAuto(name, calls) for name in ("AutoModelForCausalLM", *classes)}
    attributes.update(AutoTokenizer=FakeAuto("AutoTokenizer", calls, FakeTokenizer()), AutoProcessor=FakeAuto("AutoProcessor", calls),
                      BitsAndBytesConfig=lambda **kwargs: ("BitsAndBytesConfig", kwargs))
    return fake_module("transformers", **attributes)


def fake_libraries(calls, cuda=False, bf16=True, bitsandbytes=True, **transformers_options):
    modules = {"torch": fake_torch(cuda, bf16), "transformers": fake_transformers(calls, **transformers_options)}
    modules["bitsandbytes"] = fake_module("bitsandbytes") if bitsandbytes else None
    return mock.patch.dict(sys.modules, modules)


def model(name):
    return find_model_config(PLATFORM, name)


def loaded_with(config, **options):
    calls = []
    with fake_libraries(calls, **options):
        adapter = HuggingFaceModelAdapter(config); adapter.load()
    return adapter, calls


class ModelKindConfigTests(unittest.TestCase):
    def test_every_model_declares_kind_and_params(self):
        for config in load_model_configs(PLATFORM):
            with self.subTest(config.name):
                self.assertIn(config.kind, ("text", "multimodal"))
                self.assertGreater(config.params_b, 0)
        primary = model("primary")
        self.assertEqual(primary.kind, "multimodal")  # thẻ Hugging Face: image-text-to-text, AutoModelForMultimodalLM
        self.assertAlmostEqual(primary.params_b, 27.78, places=1)
        self.assertIn("vision", {capability.value for capability in primary.capabilities})

    def test_invalid_values_are_rejected(self):
        for value, message in (({"kind": "image"}, "kind"), ({"quantization": "nf4"}, "quantization"), ({"params_b": 0}, "params_b"), ({"params_b": "27"}, "params_b"), ({"capabilities": ["vision"]}, "multimodal")):
            with self.subTest(value), self.assertRaisesRegex(ValueError, message):
                ModelConfig.from_dict({"name": "x", "source": "org/x", **value})
        self.assertEqual(ModelConfig.from_dict({"name": "x", "source": "org/x"}).kind, "text")


class ModelLoadingTests(unittest.TestCase):
    def test_multimodal_uses_processor_and_multimodal_class(self):
        adapter, calls = loaded_with(model("primary"))
        self.assertEqual([call[0] for call in calls], ["AutoProcessor", "AutoModelForMultimodalLM"])
        self.assertEqual(calls[1][2]["dtype"], "torch.bfloat16")
        self.assertNotIn("quantization_config", calls[1][2])
        self.assertEqual(adapter.model.loaded_by, "AutoModelForMultimodalLM")

    def test_multimodal_falls_back_to_image_text_to_text(self):
        _, calls = loaded_with(model("primary"), classes=("AutoModelForImageTextToText",))
        self.assertEqual(calls[1][0], "AutoModelForImageTextToText")
        with self.assertRaisesRegex(RuntimeError, "transformers"): loaded_with(model("primary"), classes=())

    def test_text_model_uses_tokenizer_and_causal_lm(self):
        _, calls = loaded_with(model("smoke"))
        self.assertEqual([call[0] for call in calls], ["AutoTokenizer", "AutoModelForCausalLM"])


class QuantizationTests(unittest.TestCase):
    def test_4bit_and_8bit_pass_bitsandbytes_config(self):
        _, calls = loaded_with(replace(model("primary"), quantization="4bit"))
        name, settings = calls[1][2]["quantization_config"]
        self.assertEqual(name, "BitsAndBytesConfig")
        self.assertEqual(settings, {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True, "bnb_4bit_compute_dtype": "torch.bfloat16"})
        _, calls = loaded_with(replace(model("coder"), quantization="8bit"))
        self.assertEqual(calls[1][2]["quantization_config"], ("BitsAndBytesConfig", {"load_in_8bit": True}))

    def test_missing_bitsandbytes_is_reported(self):
        with self.assertRaisesRegex(RuntimeError, "bitsandbytes"): loaded_with(replace(model("smoke"), quantization="4bit"), bitsandbytes=False)


class DtypeFallbackTests(unittest.TestCase):
    def test_gpu_without_bf16_switches_to_fp16_with_warning(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.assertEqual(resolve_torch_dtype(fake_torch(cuda=True, bf16=False), "bfloat16"), "torch.float16")
        self.assertIn("Cảnh báo", stderr.getvalue())

    def test_bf16_is_kept_when_supported_or_on_cpu(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.assertEqual(resolve_torch_dtype(fake_torch(cuda=True, bf16=True), "bfloat16"), "torch.bfloat16")
            self.assertEqual(resolve_torch_dtype(fake_torch(cuda=False), "bfloat16"), "torch.bfloat16")
            self.assertEqual(resolve_torch_dtype(fake_torch(cuda=True, bf16=False), "float32"), "torch.float32")
        self.assertEqual(stderr.getvalue(), "")
        with self.assertRaisesRegex(ValueError, "dtype"): resolve_torch_dtype(fake_torch(), "int4")

    def test_quantized_compute_dtype_follows_fallback(self):
        with contextlib.redirect_stderr(io.StringIO()):
            _, calls = loaded_with(replace(model("primary"), quantization="4bit"), cuda=True, bf16=False)
        self.assertEqual(calls[1][2]["dtype"], "torch.float16")
        self.assertEqual(calls[1][2]["quantization_config"][1]["bnb_4bit_compute_dtype"], "torch.float16")


class FakeInputs(dict):
    def to(self, device): return self


class FakeProcessor:
    def __init__(self): self.calls = []
    def apply_chat_template(self, conversation, **kwargs):
        self.calls.append((conversation, kwargs))
        return FakeInputs(input_ids=types.SimpleNamespace(shape=(1, 3)))
    def decode(self, tokens, skip_special_tokens): return f"trả lời {tokens}"


class ImageMessageTests(unittest.TestCase):
    def test_multimodal_receives_image_and_text(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "anh.png"; image.write_bytes(b"png")
            adapter, processor = HuggingFaceModelAdapter(model("primary")), FakeProcessor()
            adapter.model, adapter.tokenizer = types.SimpleNamespace(device="cpu", generate=lambda **kwargs: [[1, 2, 3, 4, 5]]), processor
            answer = adapter.generate([Message("user", "Mô tả ảnh này", images=(str(image),))])
        conversation, kwargs = processor.calls[0]
        self.assertEqual(conversation, [{"role": "user", "content": [{"type": "image", "path": str(image)}, {"type": "text", "text": "Mô tả ảnh này"}]}])
        self.assertTrue(kwargs["tokenize"] and kwargs["return_dict"] and kwargs["add_generation_prompt"])
        self.assertEqual(answer, "trả lời [4, 5]")

    def test_image_errors_are_clear(self):
        with self.assertRaisesRegex(ValueError, "chỉ nhận chữ"):
            HuggingFaceModelAdapter(model("smoke")).generate([Message("user", "xem ảnh", images=("anh.png",))])
        with self.assertRaisesRegex(FileNotFoundError, "ảnh"):
            HuggingFaceModelAdapter(model("primary")).generate([Message("user", "xem ảnh", images=("khong-co.png",))])


class QloraFinetuneTests(unittest.TestCase):
    def training_libraries(self, calls, **options):
        def prepare(model, **kwargs):
            calls.append(("prepare_model_for_kbit_training", kwargs)); return model

        class Trainer:
            def __init__(self, **kwargs): calls.append(("SFTTrainer", kwargs))
            def train(self, resume_from_checkpoint): return types.SimpleNamespace(metrics={"train_loss": 1.5})
            def save_model(self, path): calls.append(("save_model", path))

        dataset = types.SimpleNamespace(select_columns=lambda columns: dataset)
        modules = {"trl": fake_module("trl", SFTConfig=lambda **kwargs: ("SFTConfig", kwargs), SFTTrainer=Trainer),
                   "peft": fake_module("peft", LoraConfig=lambda **kwargs: ("LoraConfig", kwargs), prepare_model_for_kbit_training=prepare),
                   "datasets": fake_module("datasets", load_dataset=lambda *args, **kwargs: dataset)}
        stack = contextlib.ExitStack()
        stack.enter_context(fake_libraries(calls, **options)); stack.enter_context(mock.patch.dict(sys.modules, modules))
        return stack

    def run_train(self, calls, **options):
        from local_ai.training import finetune
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "sft.jsonl"; dataset.write_text('{"messages": []}\n', encoding="utf-8")
            config = finetune.load_finetune_config(ROOT / "configs" / "training" / "qlora_primary.json", models_config=str(PLATFORM), dataset_path=str(dataset), output_dir=str(Path(directory) / "out"))
            with self.training_libraries(calls, **options), contextlib.redirect_stderr(io.StringIO()):
                return finetune.train(config)

    def test_sample_config_is_qlora_on_primary(self):
        from local_ai.training.finetune import describe, load_finetune_config
        plan = describe(load_finetune_config(ROOT / "configs" / "training" / "qlora_primary.json", models_config=str(PLATFORM)))
        self.assertEqual((plan["method"], plan["quantization"], plan["qlora"]), ("lora", "4bit", True))
        self.assertEqual((plan["base_model"]["name"], plan["base_model"]["kind"]), ("primary", "multimodal"))

    def test_qlora_loads_4bit_prepares_kbit_then_attaches_lora(self):
        calls = []
        result = self.run_train(calls, cuda=True)
        self.assertEqual(result["status"], "completed")
        names = [call[0] for call in calls]
        load = next(call for call in calls if call[0] == "AutoModelForMultimodalLM")
        self.assertEqual(load[2]["quantization_config"][1]["load_in_4bit"], True)
        self.assertLess(names.index("AutoModelForMultimodalLM"), names.index("prepare_model_for_kbit_training"))
        self.assertLess(names.index("prepare_model_for_kbit_training"), names.index("SFTTrainer"))
        trainer = next(call[1] for call in calls if call[0] == "SFTTrainer")
        self.assertEqual(trainer["peft_config"][1]["task_type"], "CAUSAL_LM")
        self.assertIsInstance(trainer["processing_class"], FakeTokenizer)
        self.assertTrue(trainer["args"][1]["bf16"])

    def test_t4_trains_in_fp16(self):
        calls = []
        self.run_train(calls, cuda=True, bf16=False)
        args = next(call[1] for call in calls if call[0] == "SFTTrainer")["args"][1]
        self.assertEqual((args["bf16"], args["fp16"]), (False, True))

    def test_missing_bitsandbytes_or_gpu_skips(self):
        from local_ai.training import finetune
        config = finetune.load_finetune_config(ROOT / "configs" / "training" / "qlora_primary.json", models_config=str(PLATFORM))
        with self.training_libraries([], cuda=True, bitsandbytes=False):
            self.assertEqual(finetune.train(config), {"status": "skipped", "missing": ["bitsandbytes"]})
        with self.training_libraries([], cuda=False):
            self.assertEqual(finetune.train(config)["missing"], ["cuda"])
        plain = finetune.load_finetune_config(ROOT / "configs" / "training" / "qlora_primary.json", models_config=str(PLATFORM), quantization="none")
        with self.training_libraries([], cuda=True, bitsandbytes=False):
            self.assertEqual(finetune.missing_requirements(plain), [])

    def test_full_finetune_on_quantized_model_is_rejected(self):
        from local_ai.training.finetune import describe, load_finetune_config
        config = load_finetune_config(ROOT / "configs" / "training" / "qlora_primary.json", models_config=str(PLATFORM), method="full")
        with self.assertRaisesRegex(ValueError, "QLoRA"): describe(config)
        with self.assertRaisesRegex(ValueError, "quantization"): load_finetune_config(ROOT / "configs" / "training" / "qlora_primary.json", quantization="nf4")


class VramEstimateTests(unittest.TestCase):
    def test_numbers_for_27b(self):
        from local_ai.models.vram import training_gb, weights_gb
        self.assertAlmostEqual(weights_gb(27, "bf16"), 54.0)
        self.assertTrue(27 <= weights_gb(27, "8bit") <= 30)
        self.assertTrue(14 <= weights_gb(27, "4bit") <= 16)
        # Trước ghi "QLoRA 27B vừa GPU 24 GB". Số đo thật của Qwen3-4B trên T4 (M15) cho thấy công thức cũ ước tính thấp
        # khoảng 2 lần; chủ repo đồng ý sửa công thức và test này ngày 26/9. Ước tính mới vượt 24 GB, vẫn vừa GPU 48 GB.
        self.assertTrue(24 < training_gb(27, "4bit") <= 48)
        self.assertGreater(training_gb(27, "bf16"), weights_gb(27, "bf16"))

    def test_command_prints_every_model_without_loading(self):
        code = "import sys; from local_ai.models import vram; vram.main([]); print('torch' in sys.modules, 'transformers' in sys.modules)"
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, env={**os.environ, "HF_HUB_OFFLINE": "1"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ước lượng", result.stdout)
        for config in load_model_configs(PLATFORM):
            self.assertIn(f"| {config.name} | {config.kind} | {config.params_b:g} |", result.stdout)
        self.assertTrue(result.stdout.strip().endswith("False False"))

    def test_model_without_params_is_marked(self):
        from local_ai.models.vram import format_table
        self.assertIn("| x | text | ? | ? |", format_table([ModelConfig(name="x", source="org/x")]))


if __name__ == "__main__":
    unittest.main()
