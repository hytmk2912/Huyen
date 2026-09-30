"""Mốc M25: model trung gian Huihui Qwen3 14B abliterated (mục `medium`), cấu hình train `configs/training/colab_14b.json`
và ước tính VRAM của `vram.py` theo độ dài chuỗi.

Không cần mạng hay GPU: metadata Hugging Face (revision, số tham số, kiến trúc, mã băm chat template) được chụp lại ngày
29/9/2026 trong `tests/fixtures/hf_api/`, test so `platform.json` với bản chụp đó.
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import unittest
from pathlib import Path

from local_ai.config.settings import find_model_config
from local_ai.models import vram
from local_ai.models.vram import REFERENCE_SEQ_LEN, max_seq_len, training_gb, weights_gb
from local_ai.training import finetune
from local_ai.training.estimate import estimate

ROOT = Path(__file__).resolve().parent.parent
PLATFORM = ROOT / "configs" / "models" / "platform.json"
CONFIG = ROOT / "configs" / "training" / "colab_14b.json"
SNAPSHOT = json.loads((ROOT / "tests" / "fixtures" / "hf_api" / "huihui_qwen3_14b_abliterated_v2_2026-09-29.json").read_text(encoding="utf-8"))
BUDGET_GB = 16.0
LONGEST_16GB = 128  # độ dài lớn nhất (bội số của 64 token) còn vừa 16 GB theo vram.py; ghi trong cấu hình, TASKS.md và README


def vi(value: float) -> str:
    return f"{value:.1f}".replace(".", ",")


class ModelEntryTests(unittest.TestCase):
    def test_platform_entry_matches_hugging_face_metadata(self):
        model = find_model_config(PLATFORM, "medium")
        api, config = SNAPSHOT["api"], SNAPSHOT["config"]
        self.assertEqual((model.source, model.revision), (api["id"], api["sha"]))
        self.assertRegex(model.revision, r"^[0-9a-f]{40}$")
        self.assertEqual(model.params_b, round(api["safetensors_total"] / 1e9, 2))
        self.assertEqual(model.kind, "text")
        self.assertTrue(all(name.endswith("ForCausalLM") for name in config["architectures"]))  # chỉ có chữ: nạp bằng AutoModelForCausalLM
        self.assertEqual((model.backend, model.dtype, model.adapter_path), ("transformers", "bfloat16", None))
        self.assertIs(api["gated"], False)  # không phải chấp nhận điều khoản trên Hugging Face trước khi tải
        self.assertEqual((api["license"], api["base_model"]), ("apache-2.0", ["Qwen/Qwen3-14B"]))
        self.assertIn("abliterated", model.source.lower())
        params = {item.name: item.params_b for item in (find_model_config(PLATFORM, name) for name in ("light", "medium", "primary"))}
        self.assertLess(params["light"], params["medium"]); self.assertLess(params["medium"], params["primary"])  # model trung gian

    def test_chat_template_lets_trl_train_on_answers_only(self):
        if importlib.util.find_spec("trl") is None: self.skipTest("thiếu thư viện: trl")
        from trl import chat_template_utils
        template = chat_template_utils.qwen3_chat_template
        self.assertEqual(hashlib.sha256(template.encode("utf-8")).hexdigest(), SNAPSHOT["chat_template"]["sha256"])  # template ở revision đã ghim = template Qwen3 mà TRL biết
        self.assertFalse(chat_template_utils.has_generation_markers(template))  # TRL tự thay bằng bản có {% generation %}


class TrainingConfigTests(unittest.TestCase):
    def test_colab_14b_is_qlora_fp16_on_medium(self):
        config = finetune.load_finetune_config(CONFIG)
        plan = finetune.describe(config)
        self.assertEqual((plan["method"], plan["quantization"], plan["qlora"], plan["base_model"]["name"], plan["base_model"]["dtype"]), ("lora", "4bit", True, "medium", "float16"))
        self.assertEqual(plan["base_model"]["revision"], SNAPSHOT["api"]["sha"])
        self.assertEqual((config.per_device_batch_size, config.gradient_checkpointing, config.assistant_only_loss, config.require_gpu), (1, True, True, True))
        self.assertEqual((config.output_dir, config.dataset_path), (".runs/colab_14b", "data/processed/hf_sft/sft.jsonl"))
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            finetune.main(["--config", str(CONFIG), "--push-to-hub", "--hub-model-id", "nguoi-dung/huyen-14b-qlora", "--dry-run"])
        self.assertEqual(json.loads(printed.getvalue())["hub"]["hub_model_id"], "nguoi-dung/huyen-14b-qlora")


class VramTests(unittest.TestCase):
    def test_chosen_length_does_not_fit_16gb_and_longest_fitting_is_recorded(self):
        params, chosen = find_model_config(PLATFORM, "medium").params_b, finetune.load_finetune_config(CONFIG).max_length
        self.assertEqual(chosen, 1024)
        self.assertGreater(training_gb(params, "4bit", chosen), BUDGET_GB)  # ở độ dài đã chọn: không vừa 16 GB
        self.assertEqual(max_seq_len(params, "4bit", BUDGET_GB), LONGEST_16GB)
        self.assertLessEqual(training_gb(params, "4bit", LONGEST_16GB), BUDGET_GB)
        self.assertGreater(training_gb(params, "4bit", LONGEST_16GB + 64), BUDGET_GB)
        self.assertLessEqual(training_gb(params, "4bit", chosen), 24 - 5)  # GPU 24 GB (ví dụ L4) còn dư khoảng 5 GB
        comment = json.loads(CONFIG.read_text(encoding="utf-8"))["_comment"]
        for phrase in (f"khoảng {vi(training_gb(params, '4bit', chosen))} GB ở {chosen} token", "KHÔNG vừa GPU 16 GB", f"là {LONGEST_16GB} token"):
            with self.subTest(phrase): self.assertIn(phrase, comment)

    def test_seq_len_keeps_calibrated_values_at_2048(self):
        for params in (0.49, 4.02, 14.77, 27.78):
            for precision in ("4bit", "bf16"):
                with self.subTest(params=params, precision=precision):
                    old = weights_gb(params, precision) * vram.TRAINING_FACTOR + vram.kbit_overhead_gb(params, precision) + vram.CUDA_CONTEXT_GB  # công thức trước M25
                    self.assertAlmostEqual(training_gb(params, precision), old)
                    self.assertAlmostEqual(training_gb(params, precision, REFERENCE_SEQ_LEN), old)
                    lengths = [128, 512, 1024, 2048, 4096]
                    values = [training_gb(params, precision, length) for length in lengths]
                    self.assertEqual(values, sorted(values)); self.assertEqual(len(set(values)), len(values))  # dài hơn thì tốn hơn
                    self.assertGreater(training_gb(params, precision, 1), weights_gb(params, precision))  # luôn còn trọng số và phần không đổi
        self.assertAlmostEqual(training_gb(4.02, "4bit"), 9.18, places=2)  # light: số trong README không đổi
        with self.assertRaisesRegex(ValueError, "seq_len"): training_gb(4.02, "4bit", 0)

    def test_longest_length_for_other_models(self):
        self.assertEqual(max_seq_len(27.78, "4bit", BUDGET_GB), 0)  # model chính: độ dài nào cũng không vừa 16 GB
        light = max_seq_len(4.02, "4bit", 15.0)
        self.assertGreaterEqual(light, REFERENCE_SEQ_LEN)  # light vừa T4 ở 2048 token như đã chạy thật
        self.assertEqual(light % 64, 0)

    def test_vram_command_prints_length_and_longest_fitting(self):
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            vram.main(["--seq-len", "1024", "--budget-gb", "16"])
        row = next(line for line in printed.getvalue().splitlines() if line.startswith("| medium |"))
        self.assertEqual(row, f"| medium | text | 14.77 | 29.5 | 15.7 | 8.3 | {training_gb(14.77, 'bf16', 1024):.1f} | 18.6 | {LONGEST_16GB} |")
        self.assertIn("| primary | multimodal | 27.78 |", printed.getvalue()); self.assertRegex(printed.getvalue(), r"\| primary \|.*\| không vừa \|")
        self.assertIn("mỗi dòng dài tối đa 1024 token", printed.getvalue())
        for argv in (["--seq-len", "0"], ["--budget-gb", "-1"]):
            with self.subTest(argv), contextlib.redirect_stderr(io.StringIO()) as error, self.assertRaises(SystemExit):
                vram.main(argv)
            self.assertIn("phải lớn hơn 0", error.getvalue())

    def test_estimate_uses_the_length_in_the_config(self):
        plans = {name: estimate(finetune.load_finetune_config(ROOT / "configs" / "training" / f"colab_{name}.json")) for name in ("smoke", "light", "14b")}
        self.assertEqual((plans["14b"]["vram_gb"], plans["14b"]["fits"]), (round(training_gb(14.77, "4bit", 1024), 1), False))
        self.assertEqual((plans["smoke"]["vram_gb"], plans["light"]["vram_gb"]), (3.2, 9.2))  # smoke (4 × 1024 token) và light (2048 token) không đổi
        self.assertTrue(plans["smoke"]["fits"] and plans["light"]["fits"])


class ReadmeTests(unittest.TestCase):
    def test_readme_explains_the_14b_model_and_its_vram(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        section = readme.split("**Model trung gian 14B (M25).**", 1)[1].split("Chấm sau mỗi bước", 1)[0]
        params = find_model_config(PLATFORM, "medium").params_b
        for phrase in (SNAPSHOT["api"]["id"], SNAPSHOT["api"]["sha"], "configs/training/colab_14b.json", "python -m local_ai.models.vram --seq-len 1024 --budget-gb 16",
                       f"khoảng {vi(training_gb(params, '4bit', 1024))} GB", f"khoảng {vi(training_gb(params, '4bit'))} GB", f"vừa 16 GB là {LONGEST_16GB} token", "chưa đo thật"):
            with self.subTest(phrase): self.assertIn(phrase, section)
        self.assertIn(f"{params:g} tỷ tham số".replace(".", ","), section)


if __name__ == "__main__":
    unittest.main()
