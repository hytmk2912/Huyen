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
import math
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
from local_ai.models.vram import CUDA_CONTEXT_GB, TRAINING_FACTOR, training_gb, weights_gb
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
        overhead = result["proposals"]["KBIT_OVERHEAD_GB"]  # model nén: tính ngược phần thêm của QLoRA (vram.py), không phải TRAINING_FACTOR
        self.assertEqual((overhead["measured"], overhead["reliable"]), (round((7.9 - weights_gb(4.02, "4bit") * TRAINING_FACTOR) / math.sqrt(4.02), 2), True))
        self.assertNotIn("TRAINING_FACTOR", result["proposals"])
        per_token = 200.0 / (13000 / 3.4)
        self.assertAlmostEqual(result["proposals"]["token_overhead_s"]["measured"], round(per_token - 2 * 4.02 / GPUS["T4"].memory_gbps, 4))
        self.assertEqual(result["agent"], {"passed": 3, "total": 5, "success_rate": 0.6, "measured_minutes": 9.0})

    def test_small_model_vram_factor_is_flagged(self):
        result = calibrate.compare(config_without_data("smoke"), fixture("smoke_t4.json"))
        self.assertFalse(result["proposals"]["KBIT_OVERHEAD_GB"]["reliable"])
        self.assertIn("KHÔNG dùng số này", calibrate.format_comparison(result))
        unquantized = calibrate.compare(config_without_data("smoke", quantization="none"), fixture("smoke_t4.json"))
        self.assertFalse(unquantized["proposals"]["TRAINING_FACTOR"]["reliable"])  # model không nén vẫn đề xuất TRAINING_FACTOR

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


class RealSmokeMeasurementTests(unittest.TestCase):
    """Số đo THẬT đầu tiên: notebook train_colab với smoke trên Colab T4 ngày 25/9 (file so_do/smoke.json chép về fixtures)."""

    def setUp(self):
        self.real = fixture("that_smoke_t4_2026-09-25.json")

    def test_t4_train_throughput_follows_real_measurement(self):
        self.assertEqual((self.real["gpu"], self.real["measurements"]["end_step"]), ("Tesla T4", 125))
        light = fixture("that_light_t4_2026-09-25.json")["measurements"]
        # Hằng số lấy theo 2 lần đo thật (smoke 5,36; light 5,17 TFLOPS): nằm giữa hai số đo, ước tính của mỗi model lệch dưới 5%.
        light_tflops = 2 * light["params_b"] * 1e9 * light["num_tokens"] * 3 / light["train_seconds"] / 1e12  # batch 1: không có phần đệm
        self.assertTrue(light_tflops <= GPUS["T4"].train_tflops <= self.real["train"]["effective_tflops"])
        result = calibrate.compare(config_without_data("smoke"), self.real["measurements"])  # tính lại từ số đo gốc, không lấy số đã làm tròn
        self.assertAlmostEqual(result["proposals"]["train_tflops"]["measured"], self.real["train"]["effective_tflops"], delta=0.05)
        plan = estimate(config_without_data("smoke"), rows=2000)
        self.assertLess(abs(plan["train_minutes"] - self.real["train"]["measured_minutes"]) / self.real["train"]["measured_minutes"], 0.05)
        self.assertFalse(result["proposals"]["KBIT_OVERHEAD_GB"]["reliable"])  # không sửa vram.py theo smoke (model quá nhỏ)

    def test_old_eval_reports_include_load_time_so_overhead_is_not_used(self):
        report = {"status": "completed", "model": "smoke", "passed": 16, "cases": 30, "duration_s": 132.0, "output_chars": 10200}
        old = calibrate.compare(config_without_data("smoke"), self.real["measurements"], {"before": report})
        self.assertFalse(old["proposals"]["token_overhead_s"]["reliable"])
        self.assertIn("tính cả thời gian tải và nạp model", calibrate.format_comparison(old))
        new = calibrate.compare(config_without_data("smoke"), self.real["measurements"], {"before": {**report, "load_s": 20.0}})
        self.assertTrue(new["proposals"]["token_overhead_s"]["reliable"]); self.assertNotIn("note", new["proposals"]["token_overhead_s"])
        self.assertEqual(GPUS["T4"].token_overhead_s, self.real["proposals"]["token_overhead_s"]["current"])  # chưa sửa theo số đo lẫn thời gian nạp

    def test_eval_loads_model_before_timing(self):
        cases = [case for case in load_cases() if case.scoring != "python_tests"][:3]
        clock = types.SimpleNamespace(now=0.0)
        fake_time = types.SimpleNamespace(monotonic=lambda: clock.now)

        class LazyModel:
            name, model, loads = "nap-luoi", None, 0
            def load(self): self.loads += 1; clock.now += 40.0; self.model = object()  # tải và nạp model mất 40 giây
            def generate(self, messages):
                if self.model is None: self.load()
                clock.now += 2.0; return "trả lời"

        from local_ai.evaluation import suite
        with mock.patch.object(suite, "time", fake_time):
            lazy = LazyModel(); report = run_eval(lazy, cases)
        self.assertEqual((lazy.loads, report["load_s"], report["duration_s"]), (1, 40.0, 6.0))

        class Broken(LazyModel):
            def load(self): raise OSError("hết bộ nhớ GPU")
        failed = run_eval(Broken(), cases)
        self.assertEqual((failed["status"], failed["answered"]), ("error", 0)); self.assertIn("Model lỗi khi nạp", failed["error"])

    def test_readme_has_real_measurement_table(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        table = readme.split("**Số đo thật trên Colab**", 1)[1].split("\n\n", 2)[1]
        smoke = next(line for line in table.splitlines() if line.startswith("| Train |"))
        self.assertIn(f"trong {self.real['train']['measured_minutes']:.1f} phút".replace(".", ","), smoke)
        before, after = self.real["eval"]["before"], self.real["eval"]["after"]
        for phrase in (f"{self.real['vram']['measured_gb']:.1f} GB".replace(".", ","), f"{before['passed']}/{before['cases']} → {after['passed']}/{after['cases']}"):
            with self.subTest(phrase): self.assertIn(phrase, table)


class RealLightMeasurementTests(unittest.TestCase):
    """Số đo THẬT của light trên Colab T4 ngày 25/9: Colab ngắt sau bước 30, lần chạy tiếp train nốt 95 bước."""

    def setUp(self):
        self.real = fixture("that_light_t4_2026-09-25.json")
        self.measured = self.real["measurements"]

    def test_resumed_run_counts_only_its_own_steps_and_tokens(self):
        self.assertEqual((self.measured["gpu"], self.measured["start_step"], self.measured["end_step"]), ("Tesla T4", 30, 125))
        tokens_per_step = self.measured["num_tokens"] / (self.measured["end_step"] - self.measured["start_step"])
        self.assertLess(abs(tokens_per_step - 16 * 558) / (16 * 558), 0.05)  # khớp số token mỗi dòng đo mẫu (cắt ở 2048, batch 1)

    def test_train_estimate_matches_real_run(self):
        result = calibrate.compare(config_without_data("light"), self.measured, {"before": self.real["eval_before"]}, max_new_tokens=512)
        train = result["train"]
        self.assertEqual(train["steps"], 95)
        self.assertLess(abs(train["estimated_minutes"] - train["measured_minutes"]) / train["measured_minutes"], 0.05)
        before = result["eval"]["before"]
        self.assertLess(abs(before["estimated_max_minutes"] - before["measured_minutes"]) / before["measured_minutes"], 0.05)
        self.assertFalse(result["proposals"]["token_overhead_s"]["reliable"])  # báo cáo cũ tính cả thời gian nạp model

    def test_qlora_vram_formula_follows_light(self):
        # Chủ repo đồng ý sửa công thức VRAM ngày 26/9: phần thêm của QLoRA lấy theo số đo của light.
        result = calibrate.compare(config_without_data("light"), self.measured)
        overhead = result["proposals"]["KBIT_OVERHEAD_GB"]
        self.assertTrue(overhead["reliable"])
        self.assertAlmostEqual(overhead["current"], overhead["measured"], delta=0.02)
        # Ước tính = số đo torch + 1 GB CUDA context (torch không đếm phần này): hơi cao hơn số đo, không thấp hơn.
        light_gb, measured_gb = training_gb(4.02, "4bit"), self.measured["peak_reserved_gb"]
        self.assertTrue(measured_gb <= light_gb <= measured_gb + CUDA_CONTEXT_GB + 0.05)
        smoke = fixture("that_smoke_t4_2026-09-25.json")["measurements"]
        smoke_gb = training_gb(smoke["params_b"], "4bit")
        self.assertTrue(smoke["peak_reserved_gb"] <= smoke_gb <= smoke["peak_reserved_gb"] * 1.3)  # smoke không dùng để hiệu chỉnh, chỉ để kiểm tra
        self.assertLess(result["vram"]["estimated_gb"], GPUS["T4"].memory_gb)

    def test_readme_table_has_light_column(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        table = readme.split("**Số đo thật trên Colab**", 1)[1].split("\n\n", 2)[1]
        before, after = self.real["eval_before"], self.real["eval_after"]
        tool = (before["groups"]["tool_use"], after["groups"]["tool_use"])
        score = f"{before['passed']}/{before['cases']} → {after['passed']}/{after['cases']}; riêng tool_use {tool[0]['passed']}/{tool[0]['cases']} → {tool[1]['passed']}/{tool[1]['cases']}"
        for phrase in ("`light` (25/9/2026)", "66,7 phút", "8,2 GB (ước tính cũ 3,8 GB, công thức mới 9,2 GB)", score, f"sau: {after['measured_minutes']:.1f} phút".replace(".", ",")):
            with self.subTest(phrase): self.assertIn(phrase, table)
        self.assertIn("Vì sao tool_use tụt sau khi train", readme)  # có đề xuất cách giữ khả năng dùng công cụ


class RealAgentMeasurementTests(unittest.TestCase):
    """Số đo THẬT của agent_colab (Ollama + qwen3:4b trên Colab T4, 26/9), chép từ ảnh chụp Bước 7 của chủ repo."""

    def setUp(self):
        self.real = fixture("that_agent_t4_2026-09-26.json")

    def test_agent_result_matches_task_set(self):
        tasks = {task.id: task for task in load_tasks()}
        measured = list(tasks)[:5]  # lần chạy thật 26/9 có 5 nhiệm vụ; M19 thêm nhiệm vụ thứ 6 (thu-lai-khi-bi-tu-choi) sau lần đo này
        self.assertEqual([item["id"] for item in self.real["tasks"]], measured)
        self.assertEqual((self.real["passed"], self.real["total"]), (sum(item["passed"] for item in self.real["tasks"]), len(measured)))
        failed = [item for item in self.real["tasks"] if not item["passed"]]
        self.assertEqual([item["id"] for item in failed], ["tong-cot-csv"])
        self.assertIn(failed[0]["expected"], tasks["tong-cot-csv"].expected)
        self.assertNotIn(failed[0]["expected"], failed[0]["answer"])

    def test_calibrate_table_and_readme_show_agent(self):
        result = calibrate.compare(config_without_data("light"), fixture("that_light_t4_2026-09-25.json")["measurements"], agent_report=self.real)
        self.assertEqual(result["agent"], {"passed": 4, "total": 5, "success_rate": 0.8, "measured_minutes": 11.1})
        self.assertIn("| Agent: nhiệm vụ đạt | | 4/5 (11,1 phút) |", calibrate.format_comparison(result))
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        table = readme.split("**Số đo thật trên Colab**", 1)[1].split("\n\n", 2)[1]
        self.assertIn("| Nhiệm vụ agent | — | — | 4/5 (80%) trong 11,1 phút; không đạt: `tong-cot-csv` |", table)
        self.assertNotIn("chưa chạy", table)  # đủ số đo của cả smoke, light và agent


class ZeroStepRerunTests(unittest.TestCase):
    """Chạy lại notebook khi checkpoint đã đủ bước (light ngày 26/9: train_seconds 0,0073): không ghi đè số đo của lần đã train."""

    def rerun(self, local_previous=None, hub_previous=None):
        class Trainer:
            def __init__(self, **kwargs): self.state = types.SimpleNamespace(log_history=[], global_step=125)
            def train(self, resume_from_checkpoint): return types.SimpleNamespace(global_step=125, metrics={"train_runtime": 0.0073})
            def save_model(self, path): pass

        class Auto:
            @staticmethod
            def from_pretrained(source, **kwargs):
                return types.SimpleNamespace(pad_token=None, eos_token="e", save_pretrained=lambda path: None, config=types.SimpleNamespace())

        cuda = types.SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: False, get_device_name=lambda index: "Tesla T4",
                                     max_memory_allocated=lambda: 4.23e9, max_memory_reserved=lambda: 4.68e9)
        dataset = types.SimpleNamespace(select_columns=lambda columns: dataset)
        modules = {"torch": fake_module("torch", bfloat16="bf16", float16="fp16", float32="fp32", cuda=cuda),
                   "transformers": fake_module("transformers", AutoTokenizer=Auto, AutoModelForCausalLM=Auto, BitsAndBytesConfig=lambda **kwargs: kwargs),
                   "bitsandbytes": fake_module("bitsandbytes"), "trl": fake_module("trl", SFTConfig=lambda **kwargs: kwargs, SFTTrainer=Trainer),
                   "peft": fake_module("peft", LoraConfig=lambda **kwargs: kwargs, prepare_model_for_kbit_training=lambda model, **kwargs: model),
                   "datasets": fake_module("datasets", load_dataset=lambda *args, **kwargs: dataset)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); output = root / "out"
            (root / "sft.jsonl").write_text('{"messages": []}\n', encoding="utf-8")
            (output / "checkpoint-125").mkdir(parents=True); (output / "checkpoint-125" / "trainer_state.json").write_text('{"global_step": 125}', encoding="utf-8")
            if local_previous: (output / MEASUREMENTS).write_text(json.dumps(local_previous), encoding="utf-8")
            remote = root / "tu_hub.json"
            if hub_previous: remote.write_text(json.dumps(hub_previous), encoding="utf-8")
            config = load_finetune_config(CONFIGS["light"], dataset_path=str(root / "sft.jsonl"), output_dir=str(output), push_to_hub=True, hub_model_id="nguoi-dung/huyen-light-qlora")
            with mock.patch.dict(sys.modules, modules), mock.patch("local_ai.training.finetune.download_file", lambda repo, name: remote if hub_previous else None), contextlib.redirect_stderr(io.StringIO()):
                result = train(config)
            return result["measurements"], json.loads((output / MEASUREMENTS).read_text(encoding="utf-8"))

    def test_keeps_measurements_of_the_run_that_trained(self):
        real = fixture("that_light_t4_2026-09-25.json")["measurements"]
        for label, options in (("trên máy", {"local_previous": real}), ("trên Hub", {"hub_previous": real})):
            with self.subTest(label):
                returned, written = self.rerun(**options)
                self.assertEqual((returned, written), (real, real))
        returned, written = self.rerun()  # chưa có số đo nào: ghi số đo 0 bước như trước
        self.assertEqual((written["start_step"], written["end_step"], written["peak_reserved_gb"]), (125, 125, 4.68))

    def test_calibrate_does_not_use_vram_of_a_zero_step_run(self):
        zero = {**fixture("that_light_t4_2026-09-25.json")["measurements"], "start_step": 125, "end_step": 125, "train_seconds": 0.0073, "peak_reserved_gb": 4.68}
        result = calibrate.compare(config_without_data("light"), zero)
        self.assertIsNone(result["train"])
        self.assertNotIn("KBIT_OVERHEAD_GB", result["proposals"])
        self.assertIn("không train bước nào", calibrate.format_comparison(result))


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
