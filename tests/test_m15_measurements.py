"""Mốc M15 (phần làm được khi chưa có số đo thật): lệnh train ghi measurements.json, eval và agent ghi thời gian,
lệnh calibrate so số đo với ước tính và đề xuất hằng số, notebook train có ô số đo ở cuối.

Số đo trong tests/fixtures/measurements/ là SỐ ĐO MẪU, không phải số đo thật. Phần train thật dùng model tí hon của test M6
trên CPU (tự bỏ qua nếu thiếu thư viện); không cần mạng hay GPU.
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

from local_ai.agents.tasks import load_tasks, format_report, run_tasks
from local_ai.evaluation.suite import load_cases, run_eval
from local_ai.models.router import ScriptedModelAdapter
from local_ai.models.vram import weights_gb
from local_ai.training import calibrate
from local_ai.training.estimate import GPUS, estimate
from local_ai.training.finetune import MEASUREMENTS, load_finetune_config, train

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).parent / "fixtures" / "measurements"
CONFIGS = {model: ROOT / "configs" / "training" / f"colab_{model}.json" for model in ("smoke", "light")}
LIBRARIES = ("torch", "transformers", "trl", "peft", "datasets", "tokenizers")
MISSING = [name for name in LIBRARIES if importlib.util.find_spec(name) is None]


def fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def config_without_data(model, **overrides):
    """Cấu hình Colab nhưng trỏ tới file dữ liệu không có, để ước tính dùng số đo mẫu (không phụ thuộc máy đang chạy test)."""
    return load_finetune_config(CONFIGS[model], dataset_path="khong-co/sft.jsonl", **overrides)


def fake_module(name, **attributes):
    module = types.ModuleType(name); module.__spec__ = importlib.machinery.ModuleSpec(name, None)
    for key, value in attributes.items(): setattr(module, key, value)
    return module


class CalibrateTests(unittest.TestCase):
    def test_proposals_follow_estimate_formulas(self):
        config, measured = config_without_data("light"), fixture("light_t4_resumed.json")
        result = calibrate.compare(config, measured, {"before": fixture("eval_before.json")}, fixture("agent.json"), max_new_tokens=512)
        plan = estimate(config, max_new_tokens=512)
        steps = measured["end_step"] - measured["start_step"]  # lần chạy tiếp sau khi Colab ngắt: chỉ tính các bước của lần này
        tflops = 2 * 4.02e9 * plan["tokens_per_step"] * steps * 3 / measured["train_seconds"] / 1e12
        self.assertEqual(result["train"], {"steps": 65, "estimated_minutes": round(plan["train_minutes"] * 65 / plan["steps"], 1), "measured_minutes": 50.0, "effective_tflops": round(tflops, 3)})
        self.assertEqual(result["proposals"]["train_tflops"], {"current": GPUS["T4"].train_tflops, "measured": round(tflops, 2)})
        factor = result["proposals"]["TRAINING_FACTOR"]
        self.assertEqual((factor["measured"], factor["reliable"]), (round(7.9 / weights_gb(4.02, "4bit"), 2), True))
        per_token = 200.0 / (13000 / 3.4)
        self.assertAlmostEqual(result["proposals"]["token_overhead_s"]["measured"], round(per_token - 2 * 4.02 / GPUS["T4"].memory_gbps, 4))
        self.assertEqual(result["agent"], {"passed": 3, "total": 5, "success_rate": 0.6, "measured_minutes": 9.0})

    def test_small_model_vram_factor_is_flagged(self):
        result = calibrate.compare(config_without_data("smoke"), fixture("smoke_t4.json"))
        self.assertFalse(result["proposals"]["TRAINING_FACTOR"]["reliable"])
        self.assertIn("KHÔNG dùng số này", calibrate.format_comparison(result))

    def test_unknown_gpu_only_records(self):
        result = calibrate.compare(config_without_data("smoke"), {**fixture("smoke_t4.json"), "gpu": "NVIDIA L4"})
        self.assertEqual((result["profile"], result["proposals"]), (None, {}))
        self.assertIn("chưa có hồ sơ ước tính", calibrate.format_comparison(result))
        self.assertEqual(calibrate.gpu_profile("Tesla T4"), "T4")

    def test_cli_prints_table_saves_and_pushes(self):
        calls = []

        class HfApi:
            def create_repo(self, **kwargs): calls.append(("create_repo", kwargs))
            def upload_file(self, **kwargs): calls.append(("upload_file", kwargs))

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "so_do.json"
            argv = ["--config", str(CONFIGS["smoke"]), "--measurements", str(FIXTURES / "smoke_t4.json"), "--eval-before", str(FIXTURES / "eval_before.json"),
                    "--eval-after", str(FIXTURES / "eval_after.json"), "--max-new-tokens", "256", "--output", str(output), "--push-to-hub", "--hub-model-id", "nguoi-dung/huyen-smoke-qlora"]
            with mock.patch.dict(sys.modules, {"huggingface_hub": fake_module("huggingface_hub", HfApi=HfApi)}), contextlib.redirect_stdout(io.StringIO()) as printed:
                self.assertEqual(calibrate.main(argv), 0)
            saved = json.loads(output.read_text(encoding="utf-8"))
        for phrase in ("| Train 125 bước (phút) |", "| VRAM khi train (GB) |", "| Điểm eval | | 8/30 → 11/30 |", "Hằng số đề xuất", "Đã đẩy số đo lên https://huggingface.co/nguoi-dung/huyen-smoke-qlora/blob/main/so_do/smoke.json"):
            with self.subTest(phrase): self.assertIn(phrase, printed.getvalue())
        self.assertEqual(saved["measurements"]["gpu"], "Tesla T4")
        self.assertEqual(calls[0], ("create_repo", {"repo_id": "nguoi-dung/huyen-smoke-qlora", "private": True, "exist_ok": True}))
        self.assertEqual(calls[1][1]["path_in_repo"], "so_do/smoke.json")

    def test_missing_measurements_and_dry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            with contextlib.redirect_stderr(io.StringIO()) as error:
                self.assertEqual(calibrate.main(["--config", str(CONFIGS["smoke"]), "--measurements", str(Path(directory) / "khong-co.json")]), 2)
            self.assertIn("hãy chạy lệnh train trước", error.getvalue())
            with mock.patch.dict(sys.modules, {"huggingface_hub": None}), contextlib.redirect_stdout(io.StringIO()) as printed:  # dry-run không gọi Hub
                self.assertEqual(calibrate.main(["--config", str(CONFIGS["light"]), "--dry-run", "--push-to-hub", "--hub-model-id", "nguoi-dung/huyen-light-qlora"]), 0)
        plan = json.loads(printed.getvalue())
        self.assertEqual((plan["files"]["measurements"]["path"], plan["push_to_hub"]), (f".runs/colab_light/{MEASUREMENTS}", "nguoi-dung/huyen-light-qlora"))


class ReportDurationTests(unittest.TestCase):
    def test_eval_report_has_duration_and_output_length(self):
        cases = load_cases()[:3]
        report = run_eval(ScriptedModelAdapter("mau", [case.reference or "" for case in cases]), cases)
        self.assertGreaterEqual(report["duration_s"], 0)
        self.assertEqual(report["output_chars"], sum(len(case.reference or "") for case in cases))

    def test_agent_report_has_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            report = run_tasks(load_tasks()[:2], lambda task: ScriptedModelAdapter("mau", list(task.reference)), directory)
        self.assertGreaterEqual(report["duration_s"], 0)
        self.assertTrue(all("duration_s" in item for item in report["tasks"]))
        self.assertRegex(format_report(report, "mau"), r"thời gian: \d+,?\d* phút|thời gian: \d+\.\d phút")


class FakeGpuTrainingTests(unittest.TestCase):
    def test_measurements_written_before_model_is_saved(self):
        seen = {}

        class Trainer:
            def __init__(self, **kwargs): self.state = types.SimpleNamespace(log_history=[{"loss": 1.0, "num_tokens": 900.0}, {"num_tokens": 1800.0}], global_step=0)
            def train(self, resume_from_checkpoint): return types.SimpleNamespace(global_step=8, metrics={"train_runtime": 12.5, "train_loss": 1.0})
            def save_model(self, path): seen["at_save"] = json.loads((Path(path).parent / MEASUREMENTS).read_text(encoding="utf-8"))

        class Auto:
            @staticmethod
            def from_pretrained(source, **kwargs):
                return types.SimpleNamespace(pad_token=None, eos_token="e", save_pretrained=lambda path: None, config=types.SimpleNamespace())

        cuda = types.SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: False, get_device_name=lambda index: "Tesla T4",
                                     max_memory_allocated=lambda: 3.21e9, max_memory_reserved=lambda: 4.5e9)
        dataset = types.SimpleNamespace(select_columns=lambda columns: dataset)
        modules = {"torch": fake_module("torch", bfloat16="bf16", float16="fp16", float32="fp32", cuda=cuda),
                   "transformers": fake_module("transformers", AutoTokenizer=Auto, AutoModelForCausalLM=Auto, BitsAndBytesConfig=lambda **kwargs: kwargs),
                   "bitsandbytes": fake_module("bitsandbytes"), "trl": fake_module("trl", SFTConfig=lambda **kwargs: kwargs, SFTTrainer=Trainer),
                   "peft": fake_module("peft", LoraConfig=lambda **kwargs: kwargs, prepare_model_for_kbit_training=lambda model, **kwargs: model),
                   "datasets": fake_module("datasets", load_dataset=lambda *args, **kwargs: dataset)}
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "sft.jsonl"; data.write_text('{"messages": []}\n', encoding="utf-8")
            config = load_finetune_config(CONFIGS["light"], dataset_path=str(data), output_dir=str(Path(directory) / "out"))
            with mock.patch.dict(sys.modules, modules), contextlib.redirect_stderr(io.StringIO()):
                result = train(config)
        measured = result["measurements"]
        self.assertEqual(seen["at_save"], measured)  # file có trước khi lưu/đẩy adapter
        self.assertEqual({key: measured[key] for key in ("model", "gpu", "start_step", "end_step", "train_seconds", "num_tokens", "peak_vram_gb", "peak_reserved_gb", "quantization", "max_length")},
                         {"model": "light", "gpu": "Tesla T4", "start_step": 0, "end_step": 8, "train_seconds": 12.5, "num_tokens": 1800.0, "peak_vram_gb": 3.21, "peak_reserved_gb": 4.5, "quantization": "4bit", "max_length": 2048})


@unittest.skipIf(MISSING, f"thiếu thư viện: {', '.join(MISSING)}")
class RealCpuTrainingTests(unittest.TestCase):
    """Train thật 2 bước rồi chạy tiếp 1 bước trên CPU với model tí hon (dùng lại cách tạo model của test M6)."""

    def test_real_run_records_measurements_and_resume_step(self):
        spec = importlib.util.spec_from_file_location("m6_helpers", Path(__file__).with_name("test_m6_cpu_pipeline.py"))
        m6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m6)
        from local_ai.models.adapters import HuggingFaceModelAdapter, ModelConfig
        from local_ai.training.finetune import FinetuneConfig, LoraSettings

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, m6.OFFLINE):
            root = Path(directory)
            rows = [{"messages": [{"role": "user", "content": f"Câu hỏi số {index}: hãy trả lời ngắn."}, {"role": "assistant", "content": f"Đây là câu trả lời số {index}."}]} for index in range(8)]
            (root / "sft.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            cases = load_cases()[:2]
            m6.make_tiny_model(root / "tiny", [message["content"] for row in rows for message in row["messages"]] + [case.prompt for case in cases])
            entry = {"name": "tiny", "source": str(root / "tiny"), "dtype": "float32", "device_map": None, "offline": True, "kind": "text", "params_b": 0.0001, "capabilities": ["chat", "reasoning"], "max_new_tokens": 4}
            (root / "models.json").write_text(json.dumps({"models": [entry]}), encoding="utf-8")
            base = dict(base_model="tiny", models_config=str(root / "models.json"), dataset_path=str(root / "sft.jsonl"), output_dir=str(root / "run"), method="lora",
                        lora=LoraSettings(r=4, alpha=8, dropout=0.0, target_modules="all-linear"), seed=0, per_device_batch_size=2, gradient_accumulation_steps=1,
                        max_length=64, gradient_checkpointing=False, logging_steps=1, save_steps=1, require_gpu=False)
            with contextlib.redirect_stderr(io.StringIO()):
                first = train(FinetuneConfig(**base, max_steps=2))
                second = train(FinetuneConfig(**base, max_steps=3))  # chạy tiếp từ checkpoint-2, như khi Colab ngắt rồi chạy lại
                report = run_eval(HuggingFaceModelAdapter(ModelConfig.from_dict({**entry, "adapter_path": second["output"]})), cases)
            written = json.loads((root / "run" / MEASUREMENTS).read_text(encoding="utf-8"))
            result = calibrate.compare(FinetuneConfig(**base, max_steps=3), written, {"after": report})
        self.assertEqual((first["measurements"]["start_step"], first["measurements"]["end_step"]), (0, 2))
        self.assertEqual((written["start_step"], written["end_step"], written["gpu"], written["peak_vram_gb"]), (2, 3, "cpu", None))
        self.assertGreater(written["train_seconds"], 0)
        self.assertGreater(written["num_tokens"] or 0, 0)  # TRL ghi số token thật vào log
        self.assertLess(written["num_tokens"], first["measurements"]["num_tokens"])  # chỉ đếm token của lần chạy tiếp (1 bước), không lấy số của lần trước (2 bước)
        self.assertGreater(report["duration_s"], 0)
        self.assertEqual((result["train"]["steps"], result["profile"], result["proposals"]), (1, None, {}))  # CPU không có hồ sơ: chỉ ghi số đo


if __name__ == "__main__":
    unittest.main()
