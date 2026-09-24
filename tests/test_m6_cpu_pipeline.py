"""Mốc M6: chạy thật trọn chuỗi trên CPU với model tí hon, không tải gì từ mạng.

fixture preset → sft.jsonl → LoRA đúng 2 bước (require_gpu=false) → lưu adapter → nạp lại → eval.
Model (Llama 2 lớp, hidden 64) và tokenizer (BPE byte-level, có chat_template) được tạo ngay trong test.
Máy thiếu torch, transformers, trl, peft hoặc datasets thì test tự bỏ qua.
"""
import importlib.util
import json
import math
import os
import sys
import tempfile
import time
import unittest
from itertools import chain
from pathlib import Path
from unittest import mock

LIBRARIES = ("torch", "transformers", "trl", "peft", "datasets")
MISSING = [name for name in LIBRARIES if importlib.util.find_spec(name) is None]
FIXTURES = Path(__file__).parent / "fixtures" / "presets"
OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
CHAT_TEMPLATE = ("{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}"
                 "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


def make_tiny_model(directory: Path, texts) -> None:
    """Tạo tokenizer BPE byte-level (đọc được mọi chữ, kể cả tiếng Việt) và model Llama 2 lớp khởi tạo ngẫu nhiên."""
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False); tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=400, special_tokens=["<|pad|>", "<|im_start|>", "<|im_end|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token="<|im_end|>", pad_token="<|pad|>")
    fast.chat_template = CHAT_TEMPLATE
    fast.save_pretrained(directory)
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=len(fast), hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=512, pad_token_id=fast.pad_token_id, eos_token_id=fast.eos_token_id, bos_token_id=None)
    LlamaForCausalLM(config).save_pretrained(directory)


@unittest.skipIf(MISSING, f"thiếu thư viện: {', '.join(MISSING)}")
class CpuPipelineTests(unittest.TestCase):
    def test_fixture_to_lora_to_reload_to_eval(self):
        from local_ai.data.core import load_records
        from local_ai.data.hub import load_preset, prepare_hf_mix
        from local_ai.evaluation.suite import load_cases, run_eval, write_reports
        from local_ai.models.adapters import HuggingFaceModelAdapter, ModelConfig
        from local_ai.training.finetune import FinetuneConfig, LoraSettings, train

        start = time.monotonic()
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, OFFLINE):
            root = Path(directory)
            # 1. Dữ liệu: fixture của 3 preset → một sft.jsonl (loader giả, không mạng).
            names = {load_preset(name)["hf_dataset"]["name"]: name for name in ("code", "reasoning", "vietnamese")}
            loader = lambda name, subset=None, **kwargs: iter(load_records(FIXTURES / f"{names[name]}.jsonl"))
            manifest = prepare_hf_mix({"code": 0.5, "reasoning": 0.25, "vietnamese": 0.25}, root / "data", loader, total=8)
            sft = load_records(root / "data" / "sft.jsonl")
            self.assertEqual(len(sft), 8)

            # 2. Model tí hon + danh sách model trỏ tới thư mục cục bộ.
            cases = load_cases()
            texts = [message["content"] for row in sft for message in row["messages"]] + [case.prompt for case in cases]
            make_tiny_model(root / "tiny", texts)
            model_entry = {"name": "tiny", "source": str(root / "tiny"), "dtype": "float32", "device_map": None, "offline": True, "kind": "text", "params_b": 0.0001, "capabilities": ["chat", "reasoning"], "max_new_tokens": 8}
            (root / "models.json").write_text(json.dumps({"models": [model_entry]}), encoding="utf-8")

            # 3. LoRA đúng 2 bước trên CPU.
            config = FinetuneConfig(base_model="tiny", models_config=str(root / "models.json"), dataset_path=str(root / "data" / "sft.jsonl"), output_dir=str(root / "run"),
                                    method="lora", lora=LoraSettings(r=4, alpha=8, dropout=0.0, target_modules="all-linear"),
                                    seed=0, max_steps=2, per_device_batch_size=2, gradient_accumulation_steps=1, max_length=128, gradient_checkpointing=False, logging_steps=1, save_steps=1000, require_gpu=False)
            result = train(config)
            self.assertEqual(result["status"], "completed", result)
            self.assertEqual(result["steps"], 2)
            self.assertTrue(math.isfinite(result["metrics"]["train_loss"]))
            adapter = Path(result["output"])
            self.assertTrue((adapter / "adapter_config.json").is_file())
            self.assertTrue(any(adapter.glob("adapter_model.*")))

            # 4. Nạp lại model gốc + adapter rồi chạy eval (điểm thấp là bình thường).
            reloaded = HuggingFaceModelAdapter(ModelConfig.from_dict({**model_entry, "adapter_path": str(adapter)}))
            report = run_eval(reloaded, cases)
            self.assertEqual(report["status"], "completed", report.get("error"))
            self.assertEqual(report["cases"], len(cases))
            self.assertEqual(type(reloaded.model).__name__, "PeftModelForCausalLM")
            json_path, _ = write_reports(report, root / "eval")
            self.assertTrue(json_path.is_file())
            elapsed = time.monotonic() - start
            print(f"\n[M6] train_loss={result['metrics']['train_loss']:.4f}, eval {report['passed']}/{report['cases']}, {elapsed:.1f} giây", file=sys.stderr)


class AdapterPathTests(unittest.TestCase):
    """Nạp LoRA adapter qua `adapter_path`, kiểm tra bằng module giả (chạy được cả khi thiếu thư viện)."""

    def test_adapter_is_loaded_on_top_of_base_model(self):
        import importlib.machinery
        import types
        from local_ai.models.adapters import ModelConfig, load_pretrained

        def module(name, **attributes):
            item = types.ModuleType(name); item.__spec__ = importlib.machinery.ModuleSpec(name, None)
            for key, value in attributes.items(): setattr(item, key, value)
            return item

        auto = types.SimpleNamespace(from_pretrained=lambda source, **kwargs: ("base", source))
        peft = module("peft", PeftModel=types.SimpleNamespace(from_pretrained=lambda model, path: ("lora", model, path)))
        fakes = {"torch": module("torch", float32="f32", bfloat16="bf16", float16="f16", cuda=types.SimpleNamespace(is_available=lambda: False)),
                 "transformers": module("transformers", AutoTokenizer=auto, AutoModelForCausalLM=auto), "peft": peft}
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(sys.modules, fakes):
            config = ModelConfig(name="m", source="org/m", dtype="float32", adapter_path=directory)
            model, _ = load_pretrained(config)
            self.assertEqual(model, ("lora", ("base", "org/m"), directory))
            with self.assertRaisesRegex(FileNotFoundError, "adapter"):
                load_pretrained(ModelConfig(name="m", source="org/m", dtype="float32", adapter_path=str(Path(directory) / "khong-co")))

    def test_max_new_tokens_is_validated(self):
        from local_ai.models.adapters import ModelConfig
        self.assertEqual(ModelConfig(name="m", source="org/m").max_new_tokens, 512)
        with self.assertRaisesRegex(ValueError, "max_new_tokens"): ModelConfig(name="m", source="org/m", max_new_tokens=0)


if __name__ == "__main__":
    unittest.main()
