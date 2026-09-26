"""Mốc M17: model chính (ảnh + chữ) chỉ gắn LoRA vào phần ngôn ngữ, không gắn vào phần xử lý ảnh.

Model chính là `huihui-ai/Huihui-Qwen3.8-27B-abliterated`, kiến trúc Qwen3.5 ảnh + chữ (`Qwen3_5ForConditionalGeneration`):
phần xử lý ảnh nằm ở `model.visual.*` (các khối vision và merger), phần ngôn ngữ ở `model.language_model.*`. Test dùng
model tí hon cùng kiến trúc, khởi tạo ngẫu nhiên trên CPU (tokenizer tí hon của test M6), không tải gì, không cần GPU.
"""
import contextlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_ai.config.settings import find_model_config
from local_ai.training.finetune import VISION_MODULES, describe, load_finetune_config, lora_exclude_modules

ROOT = Path(__file__).resolve().parent.parent
PLATFORM = ROOT / "configs" / "models" / "platform.json"
LIBRARIES = ("torch", "transformers", "trl", "peft", "datasets", "tokenizers")
MISSING = [name for name in LIBRARIES if importlib.util.find_spec(name) is None]


def qwen35_available() -> bool:
    if MISSING: return False
    import transformers
    return hasattr(transformers, "Qwen3_5ForConditionalGeneration")


def m6_helpers():
    spec = importlib.util.spec_from_file_location("m6_helpers", Path(__file__).with_name("test_m6_cpu_pipeline.py"))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def tiny_qwen35_config(vocab_size: int, pad_token_id=None, eos_token_id=None):
    """Cấu hình Qwen3.5 ảnh + chữ tí hon: 4 lớp ngôn ngữ (3 linear attention + 1 attention thường, như tỉ lệ của model 27B), 2 khối vision."""
    from transformers import Qwen3_5Config

    text = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16, vocab_size=vocab_size + 4,
                layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"], full_attention_interval=4,
                max_position_embeddings=256, pad_token_id=pad_token_id, eos_token_id=eos_token_id, partial_rotary_factor=0.25,
                rope_parameters={"mrope_interleaved": True, "mrope_section": [2, 2, 2], "partial_rotary_factor": 0.25, "rope_theta": 10000.0, "rope_type": "default"})
    vision = dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64, patch_size=16, spatial_merge_size=2,
                  temporal_patch_size=2, num_position_embeddings=64, deepstack_visual_indexes=[])
    return Qwen3_5Config(text_config=text, vision_config=vision, image_token_id=vocab_size, video_token_id=vocab_size + 1,
                         vision_start_token_id=vocab_size + 2, vision_end_token_id=vocab_size + 3)


def make_tiny_vl(directory: Path, texts) -> None:
    """Tokenizer tí hon của test M6 và model Qwen3.5 ảnh + chữ tí hon, lưu vào cùng thư mục."""
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    m6_helpers().make_tiny_model(directory, texts)  # tạo tokenizer (và model Llama, xoá ngay bên dưới)
    tokenizer = AutoTokenizer.from_pretrained(directory)
    for name in ("config.json", "model.safetensors", "generation_config.json"): (directory / name).unlink(missing_ok=True)
    torch.manual_seed(0)
    Qwen3_5ForConditionalGeneration(tiny_qwen35_config(len(tokenizer), tokenizer.pad_token_id, tokenizer.eos_token_id)).save_pretrained(directory)


def lora_layers(model) -> list[str]:
    return sorted({name.rsplit(".lora_", 1)[0] for name, _ in model.named_parameters() if ".lora_" in name})


def linear_layers(model, prefix: str) -> list[str]:
    import torch
    return sorted(name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear) and name.startswith(prefix))


class LoraScopeTests(unittest.TestCase):
    def test_only_multimodal_models_exclude_vision(self):
        self.assertEqual(lora_exclude_modules(find_model_config(PLATFORM, "primary")), VISION_MODULES)
        for name in ("light", "smoke", "coder"):
            with self.subTest(name): self.assertIsNone(lora_exclude_modules(find_model_config(PLATFORM, name)))

    def test_vision_pattern_matches_vision_parts_only(self):
        vision = ["model.visual.blocks.0.attn.qkv", "model.visual.merger.linear_fc2", "model.vision_tower.encoder.layers.0.self_attn.q_proj",
                  "model.multi_modal_projector.linear_1", "vision_model.encoder.layers.3.mlp.fc1"]
        language = ["model.language_model.layers.0.linear_attn.in_proj_qkv", "model.language_model.layers.3.self_attn.q_proj",
                    "model.language_model.layers.3.mlp.down_proj", "lm_head", "model.layers.0.self_attn.o_proj"]
        for name in vision:
            with self.subTest(name): self.assertTrue(re.fullmatch(VISION_MODULES, name))
        for name in language:
            with self.subTest(name): self.assertIsNone(re.fullmatch(VISION_MODULES, name))

    def test_dry_run_shows_lora_scope(self):
        primary = describe(load_finetune_config(ROOT / "configs" / "training" / "qlora_primary.json"))["lora"]
        self.assertEqual((primary["target_modules"], primary["exclude_modules"]), ("all-linear", VISION_MODULES))
        self.assertIn("không gắn vào phần xử lý ảnh", primary["note"])
        self.assertIsNone(describe(load_finetune_config(ROOT / "configs" / "training" / "colab_light.json"))["lora"]["exclude_modules"])
        self.assertIsNone(describe(load_finetune_config(ROOT / "configs" / "training" / "qlora_primary.json", method="full", quantization="none"))["lora"])
        result = subprocess.run([sys.executable, "-m", "local_ai.training.finetune", "--config", "configs/training/qlora_primary.json", "--dry-run"],
                                cwd=ROOT, capture_output=True, text=True, env={**os.environ, "HF_HUB_OFFLINE": "1"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["lora"]["exclude_modules"], VISION_MODULES)


@unittest.skipUnless(qwen35_available(), f"thiếu thư viện hoặc transformers chưa có Qwen3.5: {', '.join(MISSING)}")
class TinyQwen35Tests(unittest.TestCase):
    def test_all_linear_touches_vision_and_exclusion_fixes_it(self):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen3_5ForConditionalGeneration

        torch.manual_seed(0)
        before = get_peft_model(Qwen3_5ForConditionalGeneration(tiny_qwen35_config(64)), LoraConfig(r=4, lora_alpha=8, target_modules="all-linear", task_type="CAUSAL_LM"))
        self.assertTrue(any(name.startswith("base_model.model.model.visual.") for name in lora_layers(before)))  # lỗi cũ: LoRA gắn cả vào phần xử lý ảnh

        base = Qwen3_5ForConditionalGeneration(tiny_qwen35_config(64))
        language = [f"base_model.model.{name}" for name in linear_layers(base, "model.language_model.")]
        model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules="all-linear", exclude_modules=lora_exclude_modules(find_model_config(PLATFORM, "primary")), task_type="CAUSAL_LM"))
        layers = lora_layers(model)
        self.assertEqual(layers, language)  # mọi lớp Linear của phần ngôn ngữ, không gì khác
        self.assertTrue(any(".linear_attn." in name for name in layers) and any(".self_attn." in name for name in layers) and any(".mlp." in name for name in layers))
        self.assertFalse(any(".visual." in name or name.endswith("lm_head") for name in layers))

    def test_real_cpu_training_keeps_vision_without_lora(self):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForMultimodalLM
        from local_ai.training.finetune import FinetuneConfig, LoraSettings, train

        m6 = m6_helpers()
        rows = [{"messages": [{"role": "user", "content": f"Câu hỏi số {index}: hãy trả lời ngắn."}, {"role": "assistant", "content": f"Đây là câu trả lời số {index}."}]} for index in range(8)]
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, m6.OFFLINE), contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            (root / "sft.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            make_tiny_vl(root / "tiny", [message["content"] for row in rows for message in row["messages"]])
            entry = {"name": "tiny-vl", "source": str(root / "tiny"), "dtype": "float32", "device_map": None, "offline": True, "kind": "multimodal",
                     "params_b": 0.0003, "capabilities": ["chat", "vision"], "max_new_tokens": 4}
            (root / "models.json").write_text(json.dumps({"models": [entry]}), encoding="utf-8")
            config = FinetuneConfig(base_model="tiny-vl", models_config=str(root / "models.json"), dataset_path=str(root / "sft.jsonl"), output_dir=str(root / "run"),
                                    method="lora", lora=LoraSettings(r=4, alpha=8, dropout=0.0, target_modules="all-linear"), seed=0, per_device_batch_size=2,
                                    gradient_accumulation_steps=1, max_length=64, gradient_checkpointing=False, logging_steps=1, save_steps=1, require_gpu=False, max_steps=2)
            result = train(config)
            saved = json.loads((Path(result["output"]) / "adapter_config.json").read_text(encoding="utf-8"))
            base = AutoModelForMultimodalLM.from_pretrained(root / "tiny", dtype=torch.float32)
            language = [f"base_model.model.{name}" for name in linear_layers(base, "model.language_model.")]
            reloaded = lora_layers(PeftModel.from_pretrained(base, result["output"]))
        self.assertEqual((result["status"], result["steps"]), ("completed", 2))
        self.assertEqual(saved["exclude_modules"], VISION_MODULES)
        self.assertEqual(type(base).__name__, "Qwen3_5ForConditionalGeneration")  # cùng lớp với model chính
        self.assertEqual(reloaded, language)


if __name__ == "__main__":
    unittest.main()
