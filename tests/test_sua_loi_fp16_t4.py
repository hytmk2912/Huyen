"""Sửa lỗi Bước 8 (train) của notebook `train_colab` trên GPU T4, chủ repo gặp khi chạy `smoke` ngày 25/9:

    NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'

Nguyên nhân: TRL 1.13 (bản notebook ghim) tự đổi tham số LoRA của model nạp 4bit sang bfloat16 ngay trong SFTTrainer(...),
trong khi T4 train fp16 và GradScaler chỉ nhận gradient float32. Cách sửa: sau khi tạo SFTTrainer và trước trainer.train(),
đổi mọi tham số được train về float32 và in dtype của chúng ra log.

Test thật dùng model tí hon của test M6 trên CPU (tự bỏ qua nếu thiếu thư viện), đánh dấu giả là "đã nạp 4bit" để TRL
chạy đúng nhánh QLoRA; không cần mạng hay GPU. Lỗi CUDA gốc không tái hiện được trên CPU (CPU có hàm này cho bfloat16),
nên test kiểm tra điều kiện gây lỗi: dtype của tham số được train.
"""
import contextlib
import importlib.machinery
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from local_ai.training import finetune

ROOT = Path(__file__).resolve().parent.parent
LIBRARIES = ("torch", "transformers", "trl", "peft", "datasets", "tokenizers")
MISSING = [name for name in LIBRARIES if importlib.util.find_spec(name) is None]


def fake_module(name, **attributes):
    module = types.ModuleType(name); module.__spec__ = importlib.machinery.ModuleSpec(name, None)
    for key, value in attributes.items(): setattr(module, key, value)
    return module


class FakeParameter:
    """Tham số giả: dtype là chữ như torch giả của test M10 ("bf16", "fp16", "fp32")."""

    def __init__(self, dtype, size, requires_grad=True):
        self.dtype, self.size, self.requires_grad = dtype, size, requires_grad
        self.data = types.SimpleNamespace(to=lambda target: target)

    def __setattr__(self, name, value):
        if name == "data" and isinstance(value, str): object.__setattr__(self, "dtype", value)  # param.data = param.data.to(dtype) thì đổi dtype
        else: object.__setattr__(self, name, value)

    def numel(self): return self.size


class FakeTrainTests(unittest.TestCase):
    """Chạy `finetune.train` với thư viện giả: kiểm tra thứ tự SFTTrainer(...) → đổi float32 → trainer.train()."""

    def run_train(self, config_name, bf16_supported=False, **overrides):
        seen = {}
        parameters = [FakeParameter("bf16", 1000), FakeParameter("bf16", 24), FakeParameter("fp16", 5000, requires_grad=False)]

        class Trainer:
            def __init__(self, **kwargs): self.model = types.SimpleNamespace(parameters=lambda: iter(parameters))
            def train(self, resume_from_checkpoint):
                seen["at_train"] = [(parameter.dtype, parameter.requires_grad) for parameter in parameters]
                return types.SimpleNamespace(global_step=1, metrics={"train_loss": 1.0})
            def save_model(self, path): pass

        class Auto:
            @staticmethod
            def from_pretrained(source, **kwargs):
                return types.SimpleNamespace(pad_token=None, eos_token="e", save_pretrained=lambda path: None, config=types.SimpleNamespace())

        cuda = types.SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: bf16_supported)  # T4: không có bf16
        dataset = types.SimpleNamespace(select_columns=lambda columns: dataset)
        modules = {"torch": fake_module("torch", bfloat16="bf16", float16="fp16", float32="fp32", cuda=cuda),
                   "transformers": fake_module("transformers", AutoTokenizer=Auto, AutoModelForCausalLM=Auto, BitsAndBytesConfig=lambda **kwargs: kwargs),
                   "bitsandbytes": fake_module("bitsandbytes"), "trl": fake_module("trl", SFTConfig=lambda **kwargs: kwargs, SFTTrainer=Trainer),
                   "peft": fake_module("peft", LoraConfig=lambda **kwargs: kwargs, prepare_model_for_kbit_training=lambda model, **kwargs: model),
                   "datasets": fake_module("datasets", load_dataset=lambda *args, **kwargs: dataset)}
        log = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "sft.jsonl"; data.write_text('{"messages": []}\n', encoding="utf-8")
            config = finetune.load_finetune_config(ROOT / "configs" / "training" / config_name, dataset_path=str(data), output_dir=str(Path(directory) / "out"), **overrides)
            with mock.patch.dict(sys.modules, modules), contextlib.redirect_stderr(log):
                finetune.train(config)
        return seen["at_train"], log.getvalue()

    def test_colab_qlora_fp16_trains_float32_parameters(self):
        for name in ("colab_smoke.json", "colab_light.json"):
            with self.subTest(name):
                at_train, log = self.run_train(name)
                self.assertEqual(at_train, [("fp32", True), ("fp32", True), ("fp16", False)])  # tham số đóng băng giữ nguyên
                self.assertIn("Tham số được train: 1.024 tham số fp32; đã đổi 2 tensor bf16 sang torch.float32 trước khi train", log)

    def test_bf16_without_quantization_is_left_alone(self):
        at_train, log = self.run_train("sft.json", bf16_supported=True, quantization="none", dtype="bfloat16")
        self.assertEqual(at_train, [("bf16", True), ("bf16", True), ("fp16", False)])
        self.assertIn("Tham số được train: 1.024 tham số bf16", log); self.assertNotIn("đã đổi", log)

    def test_model_without_parameters_is_skipped(self):
        self.assertIsNone(finetune.trainable_to_float32(types.SimpleNamespace(float32="fp32"), None, cast=True))
        self.assertIsNone(finetune.trainable_to_float32(types.SimpleNamespace(float32="fp32"), types.SimpleNamespace(parameters=lambda: iter([])), cast=True))


@unittest.skipIf(MISSING, f"thiếu thư viện: {', '.join(MISSING)}")
class RealTrlQloraTests(unittest.TestCase):
    """SFTTrainer thật (TRL cùng bản với notebook) trên model tí hon được đánh dấu "đã nạp 4bit"."""

    def test_trl_casts_lora_to_bf16_and_fix_restores_float32(self):
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer

        spec = importlib.util.spec_from_file_location("m6_helpers", Path(__file__).with_name("test_m6_cpu_pipeline.py"))
        m6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m6)
        rows = [{"messages": [{"role": "user", "content": f"Câu hỏi số {index}"}, {"role": "assistant", "content": f"Trả lời số {index}"}]} for index in range(4)]
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, m6.OFFLINE), contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            m6.make_tiny_model(Path(directory) / "tiny", [message["content"] for row in rows for message in row["messages"]])
            model = AutoModelForCausalLM.from_pretrained(str(Path(directory) / "tiny"), dtype=torch.float32)
            model.is_loaded_in_4bit = True  # TRL nhận ra QLoRA qua cờ này (như model nạp bằng bitsandbytes trên T4)
            tokenizer = AutoTokenizer.from_pretrained(str(Path(directory) / "tiny"))
            args = SFTConfig(output_dir=str(Path(directory) / "run"), max_steps=1, per_device_train_batch_size=2, max_length=32, report_to=[], save_strategy="no", logging_steps=1, use_cpu=True)
            trainer = SFTTrainer(model=model, args=args, train_dataset=Dataset.from_list(rows), processing_class=tokenizer,
                                 peft_config=LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM"))
            trainable = lambda: {parameter.dtype for parameter in trainer.model.parameters() if parameter.requires_grad}
            frozen = {name: parameter.dtype for name, parameter in trainer.model.named_parameters() if not parameter.requires_grad}
            self.assertEqual(trainable(), {torch.bfloat16})  # đây là điều kiện gây lỗi trên T4 fp16
            summary = finetune.trainable_to_float32(torch, trainer.model, cast=True)
            self.assertEqual(trainable(), {torch.float32})
            self.assertEqual({name: parameter.dtype for name, parameter in trainer.model.named_parameters() if not parameter.requires_grad}, frozen)
            self.assertRegex(summary, r"^Tham số được train: [\d.]+ tham số torch\.float32; đã đổi \d+ tensor torch\.bfloat16 sang torch\.float32 trước khi train$")
            model.is_loaded_in_4bit = False  # bỏ cờ giả trước khi train: model thật không nén, không có bitsandbytes để đếm tham số 4bit
            result = trainer.train()  # optimizer tạo sau khi đổi dtype nên train được
        self.assertEqual(result.global_step, 1)
        self.assertEqual(trainable(), {torch.float32})


if __name__ == "__main__":
    unittest.main()
