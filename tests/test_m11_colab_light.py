"""Mốc M11: Colab cho model light (Qwen3-4B): cấu hình QLoRA vừa T4, notebook chọn được model, ước tính thời gian,
tự train tiếp khi Colab ngắt, hướng dẫn trên iPhone.

Không cần mạng, GPU hay model thật: lệnh của notebook chạy bằng `--dry-run`, huggingface_hub là module giả.
"""
import contextlib
import importlib.util
import io
import json
import re
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from local_ai.config.settings import find_model_config
from local_ai.evaluation.suite import load_cases
from local_ai.models.vram import training_gb
from local_ai.training import estimate as timing
from local_ai.training.finetune import load_finetune_config

ROOT = Path(__file__).resolve().parent.parent
PLATFORM = ROOT / "configs" / "models" / "platform.json"
CONFIGS = {model: ROOT / "configs" / "training" / f"colab_{model}.json" for model in ("smoke", "light")}
GUIDE = ROOT / "docs" / "TRAIN_COLAB.md"

# Dùng lại các hàm đọc notebook của test M10 (nạp theo đường dẫn để chạy được cả khi gọi `python -m unittest tests.test_m11_colab_light`).
_spec = importlib.util.spec_from_file_location("m10_helpers", Path(__file__).with_name("test_m10_colab.py"))
m10 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(m10)


def planned(model: str) -> dict:
    """Ước tính dùng trong tài liệu: 2000 dòng, số token đo mẫu (không phụ thuộc file dữ liệu có sẵn trên máy)."""
    variables = m10.notebook_variables(model)
    return timing.estimate(load_finetune_config(CONFIGS[model]), rows=2000, max_new_tokens=int(variables["MAX_NEW_TOKENS"]))


class LightConfigTests(unittest.TestCase):
    def test_light_config_is_qlora_fp16_batch_1_and_fits_t4(self):
        config = load_finetune_config(CONFIGS["light"])
        self.assertEqual((config.base_model, config.method, config.quantization, config.dtype, config.per_device_batch_size), ("light", "lora", "4bit", "float16", 1))
        self.assertTrue(config.gradient_checkpointing and config.resume is True and config.hub_private and not config.push_to_hub)
        self.assertEqual(config.dataset_path, "data/processed/hf_sft/sft.jsonl")
        light = find_model_config(PLATFORM, "light")
        self.assertLessEqual(config.max_length, 2048)  # độ dài mà ước tính VRAM của vram.py giả định
        self.assertLess(training_gb(light.params_b, "4bit"), timing.GPUS["T4"].memory_gb)
        self.assertLessEqual(config.save_steps, load_finetune_config(CONFIGS["smoke"]).save_steps)  # train lâu hơn nên lưu dày hơn

    def test_light_colab_model_points_to_light_adapter(self):
        config, light, colab = load_finetune_config(CONFIGS["light"]), find_model_config(PLATFORM, "light"), find_model_config(PLATFORM, "light-colab")
        self.assertEqual((colab.source, colab.params_b, colab.dtype, colab.adapter_path), (light.source, light.params_b, "float16", f"{config.output_dir}/adapter"))
        self.assertEqual(str(colab.max_new_tokens), m10.notebook_variables("light")["MAX_NEW_TOKENS"])


class NotebookChoiceTests(unittest.TestCase):
    def test_choice_cell_offers_smoke_and_light(self):
        cells = m10.code_cells(m10.load_notebook())
        self.assertIn('MODEL = "smoke"  # @param ["smoke", "light"]', cells["buoc-1-chon-model"])
        self.assertIn('HUB_REPO = f"{HF_USER}/huyen-{MODEL}-qlora"', cells["buoc-4-token"])  # mỗi model một repo, không lẫn checkpoint
        self.assertIn("--hub-model-id {HUB_REPO}", cells["buoc-6-uoc-tinh"])  # ô ước tính xem tiến độ trên Hub
        self.assertIn("--push-to-hub --hub-model-id {HUB_REPO}", cells["buoc-8-train"])  # checkpoint lên Hub để train tiếp
        for model, path in CONFIGS.items():
            with self.subTest(model):
                self.assertTrue(path.is_file())
                self.assertEqual(m10.notebook_variables(model)["MODEL"], model)

    def test_light_notebook_commands_run_with_dry_run(self):
        outputs = m10.run_notebook_commands(self, "light")
        (_, data), (_, estimate), (_, before), (_, train), (_, after), (_, compare), (_, push), _ = outputs  # ô cuối là số đo thật (M15)
        self.assertEqual(sum(item["rows"] for item in data["presets"].values()), 2000)
        self.assertIn("model light", estimate); self.assertRegex(estimate, r"Train: \d+ bước, khoảng \d+ phút"); self.assertNotIn("CẢNH BÁO", estimate)
        self.assertEqual((before["model"]["name"], after["model"]["name"], before["model"]["max_new_tokens"], after["model"]["max_new_tokens"]), ("light", "light-colab", 512, 512))
        self.assertEqual((train["base_model"]["name"], train["qlora"], train["base_model"]["dtype"], train["config"]["per_device_batch_size"]), ("light", True, "float16", 1))
        self.assertEqual((train["hub"]["hub_model_id"], train["hub"]["hub_strategy"], train["hub"]["resume_from_hub"]), ("nguoi-dung/huyen-light-qlora", "checkpoint", True))
        self.assertEqual((compare["before"], compare["after"]), (f"{before['output']}/report.json", f"{after['output']}/report.json"))
        self.assertEqual(push["adapter_dir"], after["model"]["adapter_path"])
        self.assertEqual(push["adapter_dir"], train["output_dir"] + "/adapter")


class EstimateTests(unittest.TestCase):
    def test_token_count_steps_and_padding(self):
        row = {"messages": [{"role": "user", "content": "x" * 34}, {"role": "assistant", "content": "y" * 68}]}
        self.assertEqual(timing.row_tokens(row, 1024), 10 + 20 + 2 * timing.TEMPLATE_TOKENS)
        self.assertEqual(timing.row_tokens(row, 16), 16)  # cắt ở max_length
        self.assertEqual(timing.padded_average([5, 10, 20], batch=1, seed=1), 35 / 3)
        self.assertEqual(timing.padded_average([5, 10, 20], batch=3, seed=1), 20)  # cả batch đệm tới dòng dài nhất
        config = load_finetune_config(CONFIGS["light"])
        self.assertEqual(timing.training_steps(config, 2000), 125)
        self.assertEqual(timing.training_steps(replace(config, max_steps=7), 2000), 7)

    def test_counts_rows_from_sft_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sft.jsonl"
            rows = [{"messages": [{"role": "user", "content": "a" * 340}]}, {"messages": [{"role": "user", "content": "b" * 34}]}]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            plan = timing.estimate(load_finetune_config(CONFIGS["light"], dataset_path=str(path)), eval_cases=30)
        self.assertEqual((plan["rows"], plan["data"], plan["steps"], plan["tokens_per_row"]), (2, "sft.jsonl", 1, (105 + 15) / 2))

    def test_light_is_slower_than_smoke_and_both_fit_t4(self):
        smoke, light = planned("smoke"), planned("light")
        self.assertTrue(smoke["fits"] and light["fits"])
        self.assertGreater(light["train_minutes"], 4 * smoke["train_minutes"])
        self.assertGreater(light["eval_minutes"], smoke["eval_minutes"])
        self.assertEqual((light["steps"], light["eval_cases"], light["data"]), (125, len(load_cases()), "đo mẫu"))  # M19: số câu chấm lấy từ bộ chấm hiện tại (38)
        too_long = timing.estimate(load_finetune_config(CONFIGS["light"], max_length=8192), rows=2000)
        self.assertFalse(too_long["fits"])
        self.assertIn("CẢNH BÁO", timing.format_estimate(too_long))

    def test_resumed_run_reports_remaining_steps(self):
        plan = timing.estimate(load_finetune_config(CONFIGS["light"]), rows=2000, trainer_state={"global_step": 60})
        self.assertEqual(plan["done_steps"], 60)
        self.assertAlmostEqual(plan["remaining_train_minutes"], plan["train_minutes"] * 65 / 125, places=0)
        self.assertIn("Đã train 60/125 bước (checkpoint trên Hugging Face)", timing.format_estimate(plan, "trên Hugging Face"))
        with tempfile.TemporaryDirectory() as directory:  # checkpoint trên máy (Colab chưa ngắt, chỉ chạy lại ô train)
            (Path(directory) / "checkpoint-20").mkdir(); (Path(directory) / "checkpoint-20" / "trainer_state.json").write_text('{"global_step": 20}', encoding="utf-8")
            self.assertEqual(timing.local_trainer_state(load_finetune_config(CONFIGS["light"], output_dir=directory))["global_step"], 20)

    def test_cli_reads_progress_from_hub_only_when_allowed(self):
        repo = "nguoi-dung/huyen-light-qlora"
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "trainer_state.json"; state.write_text('{"global_step": 40}', encoding="utf-8")
            calls = []

            def download(**kwargs):
                calls.append(kwargs); return str(state)

            config = ["--config", str(CONFIGS["light"]), "--rows", "2000", "--hub-model-id", repo]
            with m10.fake_hub([]), mock.patch.object(sys.modules["huggingface_hub"], "hf_hub_download", download, create=True), contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(timing.main(config), 0)
            self.assertEqual(calls, [{"repo_id": repo, "filename": "last-checkpoint/trainer_state.json"}])
            self.assertIn("Đã train 40/125 bước (checkpoint trên Hugging Face)", output.getvalue())

            def broken(**kwargs): raise OSError("mất mạng")
            with m10.fake_hub([]), mock.patch.object(sys.modules["huggingface_hub"], "hf_hub_download", broken, create=True), contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(timing.main(config), 0)  # chỉ để xem tiến độ: lỗi mạng không làm hỏng ô notebook
            self.assertIn("Không đọc được tiến độ trên Hugging Face (OSError)", output.getvalue())

            def missing(**kwargs): raise m10.EntryNotFoundError("chưa có checkpoint")
            with m10.fake_hub([]), mock.patch.object(sys.modules["huggingface_hub"], "hf_hub_download", missing, create=True), contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(timing.main(config), 0)
            self.assertNotIn("Đã train", output.getvalue())
        with mock.patch.dict(sys.modules, {"huggingface_hub": None}), contextlib.redirect_stdout(io.StringIO()) as output:  # --dry-run không gọi Hub
            self.assertEqual(timing.main([*config, "--dry-run"]), 0)
        self.assertIn("Train: 125 bước", output.getvalue())


class GuideTests(unittest.TestCase):
    def test_iphone_guide_covers_every_step(self):
        guide = GUIDE.read_text(encoding="utf-8")
        for phrase in ("iPhone", m10.COLAB_LINK, "Yêu cầu trang web cho máy tính", "T4 GPU", "Secrets", "`HF_TOKEN`", "Notebook access", "Run all",
                       "## Bước 6: xem kết quả", "huyen-<model>-qlora", "## Khi Colab ngắt giữa chừng", "## Lỗi hay gặp", "CUDA out of memory", "Write", "chưa chạy thử trên Colab thật"):
            with self.subTest(phrase): self.assertIn(phrase, guide)
        for text in ((ROOT / "README.md").read_text(encoding="utf-8"), "".join(m10.load_notebook()["cells"][0]["source"])):
            self.assertIn("TRAIN_COLAB.md", text)

    def test_times_in_docs_match_estimator(self):
        guide, readme = GUIDE.read_text(encoding="utf-8"), (ROOT / "README.md").read_text(encoding="utf-8")
        for model in CONFIGS:
            with self.subTest(model):
                plan = planned(model)
                total = round(plan["remaining_train_minutes"] + 2 * plan["eval_minutes"])
                row = next(line for line in guide.splitlines() if line.startswith(f"| `{model}`"))
                self.assertTrue(row.endswith(f"| khoảng {round(plan['train_minutes'])} phút | khoảng {round(plan['eval_minutes'])} phút | khoảng {total} phút |"), row)
                readme_row = next(line for line in readme.splitlines() if line.startswith(f"| `{model}`"))
                self.assertTrue(readme_row.endswith(f"| khoảng {total} phút |"), readme_row)
        light = find_model_config(PLATFORM, "light")
        self.assertIn(f"VRAM ước tính {training_gb(light.params_b, '4bit'):.1f} GB".replace(".", ","), readme)
        self.assertRegex(guide, re.escape("`light` đẩy sau mỗi ") + str(load_finetune_config(CONFIGS["light"]).save_steps) + " bước")


if __name__ == "__main__":
    unittest.main()
