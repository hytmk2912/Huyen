"""Mốc M20: dữ liệu train giữ khả năng gọi công cụ.

Khi chạy thật (M15), `light` tụt tool_use 7/8 → 3/8 sau khi train, vì dữ liệu train không có dòng gọi công cụ.
1. `hf-sft --tool-calls 0.1` trộn khoảng 10% dòng gọi công cụ tự sinh (đúng dạng JSON của câu tool_use trong bộ chấm,
   có cả câu không cần công cụ; câu hỏi và số liệu khác bộ chấm); notebook train bật mặc định.
2. Chặn gần trùng giữa dữ liệu train và bộ chấm bằng MinHash của M12; `manifest.json` ghi số dòng bị loại.
3. Chỉ tính loss trên câu trả lời (`assistant_only_loss=True`); chat template không hỗ trợ thì báo lỗi tiếng Việt trước khi train.

Không cần mạng hay GPU: loader giả đọc fixture, model tí hon khởi tạo ngẫu nhiên trên CPU.
"""
import ast
import contextlib
import hashlib
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

from local_ai.agents.loop import extract_json_object
from local_ai.config.settings import find_model_config
from local_ai.data.core import build_dataset, load_records, write_jsonl
from local_ai.data.hub import load_preset, prepare_hf_mix
from local_ai.data.quality import NearDuplicateFilter, QualityConfig, conversation, eval_conversation, jaccard, shingles
from local_ai.data.tool_calls import JSON_FORMAT, eval_numbers, eval_tool_prompts, tool_call_records, tool_call_rows
from local_ai.evaluation.suite import load_cases
from local_ai.training.finetune import FinetuneConfig, LoraSettings, check_assistant_only_loss, describe, load_finetune_config

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "data" / "eval" / "eval_v1.jsonl"
EVAL_SOURCES = [str(ROOT / "data" / "eval" / "seed_eval.jsonl"), str(EVAL)]
FIXTURES = Path(__file__).parent / "fixtures"
LIBRARIES = ("torch", "transformers", "trl", "peft", "datasets", "tokenizers")
MISSING = [name for name in LIBRARIES if importlib.util.find_spec(name) is None]
EVAL_FILES = ("notes.txt", "bao_cao.txt", "data/sales.csv", "ghi_chu/hop_nhom.md")


def helpers(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_helpers_m20", Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def record(identifier: str, question: str, answer: str) -> dict:
    return {"id": identifier, "domain": "reasoning", "task": "sft", "input": question, "expected_output": answer, "source": {"name": "thu"}, "license": {"name": "CC0-1.0"}, "dataset_version": "t"}


def fixture_loader():
    names = {load_preset(name)["hf_dataset"]["name"]: name for name in ("code", "reasoning", "vietnamese")}
    return lambda name, subset=None, **kwargs: iter(load_records(FIXTURES / "presets" / f"{names[name]}.jsonl"))


class ToolCallDataTests(unittest.TestCase):
    def setUp(self):
        self.records = tool_call_records(200)

    def test_answers_use_the_eval_json_format_and_include_no_tool_cases(self):
        eval_prompts = eval_tool_prompts()
        for language, text in JSON_FORMAT.items():  # đúng dạng JSON mà câu tool_use của bộ chấm yêu cầu
            with self.subTest(language): self.assertTrue(any(text in prompt for prompt in eval_prompts))
        kinds = {}
        for item in self.records:
            prompt, answer = item["messages"][0]["content"], item["messages"][1]["content"]
            language = item["metadata"]["language"]
            self.assertIn(JSON_FORMAT[language], prompt)
            parsed = json.loads(answer)
            self.assertEqual(set(parsed), {"tool", "arguments"}); self.assertEqual(parsed, extract_json_object(answer))  # agent đọc được
            self.assertRegex(answer, r'"tool"\s*:\s*(null|"(calculator|read_file|search)")')
            key = next(iter(parsed["arguments"]), None)
            self.assertEqual(key, {"calculator": "expression", "read_file": "path", "search": "query", None: None}[parsed["tool"]])
            if parsed["tool"] == "calculator": self.assertEqual(eval(parsed["arguments"]["expression"], {"__builtins__": {}}) is not None, True)  # biểu thức tính được
            kinds[(language, parsed["tool"])] = kinds.get((language, parsed["tool"]), 0) + 1
        self.assertEqual(set(kinds), {(language, tool) for language in ("vi", "en") for tool in ("calculator", "read_file", "search", None)})
        self.assertEqual(set(kinds.values()), {25})  # chia đều 2 ngôn ngữ × 4 loại (có câu không cần công cụ)

    def test_questions_and_numbers_differ_from_the_eval_set(self):
        cases = load_cases()
        banned = eval_numbers(eval_tool_prompts())
        eval_questions = {case.prompt for case in cases}
        threshold = NearDuplicateFilter().threshold
        for item in self.records:
            prompt = item["messages"][0]["content"]
            with self.subTest(item["id"]):
                self.assertNotIn(prompt, eval_questions)
                self.assertFalse({int(number) for number in re.findall(r"\d+", prompt)} & banned)
                self.assertFalse([name for name in EVAL_FILES if name in prompt])
                self.assertNotIn("Available tools:", prompt); self.assertNotIn("Các công cụ có sẵn:", prompt)  # lời dẫn khác bộ chấm
        question = max(jaccard(shingles(item["input"], 3), shingles(case.prompt, 3)) for item in self.records for case in cases)
        whole = max(jaccard(shingles("\n".join(conversation(item)), 3), shingles("\n".join(eval_conversation(vars(case))), 3)) for item in self.records for case in cases)
        self.assertLess(max(question, whole), threshold / 2)  # đo 26/9: tối đa khoảng 0,3

    def test_generation_is_repeatable_unique_and_uses_no_api(self):
        self.assertEqual(tool_call_records(50, seed=5), tool_call_records(50, seed=5))
        self.assertNotEqual(tool_call_records(50, seed=5), tool_call_records(50, seed=6))
        self.assertEqual(len({item["input"] for item in self.records}), len(self.records))
        tree = ast.parse((ROOT / "local_ai" / "data" / "tool_calls.py").read_text(encoding="utf-8"))
        imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        imported |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        self.assertLessEqual(imported, {"__future__", "json", "random", "re", "itertools", "pathlib", "typing", "local_ai"})  # không gọi mạng hay model
        self.assertEqual(tool_call_rows(2000, 0.1), (1800, 200))
        for share in (-0.1, 1.0, 1.5):
            with self.subTest(share), self.assertRaises(ValueError): tool_call_rows(2000, share)

    def test_hf_sft_mixes_about_ten_percent_tool_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = prepare_hf_mix({"code": 0.4, "reasoning": 0.3, "vietnamese": 0.3}, directory, fixture_loader(), total=40, tool_calls=0.1)
            sft = load_records(Path(directory) / "sft.jsonl")
        self.assertEqual(manifest["configuration"]["tool_calls"], {"share": 0.1, "rows": 4, "generator": "local_ai/data/tool_calls.py"})
        self.assertEqual(sum(item["rows"] for item in manifest["configuration"]["mix"].values()), 36)  # 40 dòng: 36 dòng preset + 4 dòng gọi công cụ
        tools = [row for row in sft if row["id"].startswith("tool-call-")]
        self.assertEqual(len(tools), 4); self.assertEqual(manifest["eval_near_duplicate"]["removed"], 0)
        self.assertEqual(sorted(str(json.loads(row["messages"][-1]["content"])["tool"]) for row in tools), ["calculator", "calculator", "read_file", "read_file"])

    def test_cli_option_and_notebook_default(self):
        command = [sys.executable, "-m", "local_ai.data", "hf-sft", "--preset", "code:0.4", "--preset", "reasoning:0.3", "--preset", "vietnamese:0.3", "--total", "2000", "--dry-run"]
        environ = {**os.environ, "HF_HUB_OFFLINE": "1"}
        plan = json.loads(subprocess.run([*command, "--tool-calls", "0.1"], cwd=ROOT, capture_output=True, text=True, env=environ, check=True).stdout)
        self.assertEqual((plan["tool_calls"], sum(item["rows"] for item in plan["presets"].values())), ({"share": 0.1, "rows": 200}, 1800))
        self.assertNotIn("tool_calls", json.loads(subprocess.run(command, cwd=ROOT, capture_output=True, text=True, env=environ, check=True).stdout))  # mặc định không trộn
        failed = subprocess.run([*command, "--tool-calls", "1"], cwd=ROOT, capture_output=True, text=True, env=environ)
        self.assertNotEqual(failed.returncode, 0); self.assertIn("[0, 1)", failed.stderr)
        cells = {cell["id"]: "".join(cell["source"]) for cell in json.loads((ROOT / "notebooks" / "train_colab.ipynb").read_text(encoding="utf-8"))["cells"]}
        self.assertIn("--total 2000 --tool-calls 0.1", cells["buoc-5-du-lieu"])


class EvalNearDuplicateTests(unittest.TestCase):
    QUESTION = "A train leaves at 14:30 and the trip takes 2 hours 45 minutes. At what time does it arrive? Use the 24-hour HH:MM format."  # câu en-reason-03 của bộ chấm

    def build(self, records, quality=None):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.jsonl"; write_jsonl(raw, records)
            manifest = build_dataset([raw], Path(directory) / "out", "t", {"formats": ["sft"]}, EVAL_SOURCES, quality)
            return manifest, load_records(Path(directory) / "out" / "sft.jsonl"), load_records(Path(directory) / "out" / "rejected.jsonl")

    def test_near_duplicates_of_eval_are_removed_and_counted(self):
        records = [record("chep-cau-hoi", self.QUESTION + " Please.", "Let me think step by step. Departure is 14:30, add 2 hours to get 16:30, then add 45 minutes: the train arrives at 17:15."),
                   record("chep-ca-dap-an", self.QUESTION.replace("A train", "The train"), "It arrives at 17:15."),
                   record("khac-han", "A bus leaves at 09:10 and the ride takes 50 minutes. When does it arrive?", "It arrives at 10:00.")]
        for quality in (None, QualityConfig.from_file()):  # chặn cả khi tắt bộ lọc chất lượng
            with self.subTest("có lọc chất lượng" if quality else "tắt lọc chất lượng"):
                manifest, kept, rejected = self.build(records, quality)
                self.assertEqual([row["id"] for row in kept], ["khac-han"])
                report = manifest["eval_near_duplicate"]
                self.assertEqual((report["removed"], report["threshold"], report["eval_records"]), (2, 0.8, len(load_cases()) + 1))
                self.assertEqual({item["id"]: (item["eval_id"], item["part"]) for item in report["examples"]}, {"chep-cau-hoi": ("en-reason-03", "câu hỏi"), "chep-ca-dap-an": ("en-reason-03", "câu hỏi")})
                reasons = {row["id"]: row["verification"] for row in rejected}
                self.assertEqual({key: value["quality_filter"] for key, value in reasons.items()}, {"chep-cau-hoi": "eval_near_duplicate", "chep-ca-dap-an": "eval_near_duplicate"})
                self.assertIn("gần trùng với câu eval en-reason-03", reasons["chep-cau-hoi"]["quality_reason"])

    def test_whole_conversation_is_compared_too(self):
        case = {item.id: item for item in load_cases()}["en-reason-06"]
        train = record("doi-mot-tu", case.prompt.replace("Ann is taller", "Ann is much taller"), case.reference)  # sửa câu hỏi một chút, giữ nguyên đáp án
        question = jaccard(shingles(train["input"], 3), shingles(case.prompt, 3))
        whole = jaccard(shingles("\n".join(conversation(train)), 3), shingles("\n".join(eval_conversation(vars(case))), 3))
        self.assertLess(question, whole)
        found = helpers_quality().find_eval_near_duplicates([train], [vars(case)], NearDuplicateFilter(threshold=round((question + whole) / 2, 3)))
        self.assertEqual((found[0][0], found[0][2]), ("en-reason-06", "cả hội thoại"))  # câu hỏi chưa đủ giống, cả hội thoại thì đủ

    def test_uses_minhash_of_m12_and_exact_overlap_still_fails(self):
        quality = helpers_quality()
        with mock.patch.object(quality.MinHash, "signature", autospec=True, side_effect=quality.MinHash.signature) as signature:
            self.build([record("khac-han", "Name three primary colors.", "Red, yellow and blue.")])
        self.assertGreater(signature.call_count, len(load_cases()))  # chữ ký MinHash của cả câu eval và dòng train
        with self.assertRaisesRegex(ValueError, "trùng với dữ liệu eval"): self.build([record("trung-han", self.QUESTION, "17:15")])

    def test_real_mix_keeps_tool_rows_and_reports_removed(self):
        rows = tool_call_records(200)
        manifest, kept, _ = self.build(rows, QualityConfig.from_file())
        self.assertEqual(manifest["eval_near_duplicate"]["removed"], 0)
        self.assertGreaterEqual(len(kept), 190)  # vài dòng tự sinh gần trùng nhau bị bộ lọc M12 loại, không phải vì giống bộ chấm


def helpers_quality():
    from local_ai.data import quality
    return quality


class AssistantOnlyLossConfigTests(unittest.TestCase):
    def test_training_configs_turn_it_on_and_it_can_be_turned_off(self):
        for path in sorted((ROOT / "configs" / "training").glob("*.json")):
            with self.subTest(path.name):
                config = load_finetune_config(path)
                self.assertIs(config.assistant_only_loss, True)
                self.assertIs(describe(config)["assistant_only_loss"], True)
        self.assertIs(FinetuneConfig(base_model="smoke", dataset_path="x.jsonl", output_dir="o").assistant_only_loss, False)  # cấu hình cũ không có khóa: như trước
        self.assertIs(FinetuneConfig.from_dict({**json.loads((ROOT / "configs" / "training" / "colab_smoke.json").read_text(encoding="utf-8")), "assistant_only_loss": False}).assistant_only_loss, False)
        with self.assertRaisesRegex(ValueError, "assistant_only_loss"): load_finetune_config(ROOT / "configs" / "training" / "colab_smoke.json", assistant_only_loss="có")

    def test_pinned_models_use_templates_trl_can_patch(self):
        if importlib.util.find_spec("trl") is None: self.skipTest("thiếu thư viện: trl")
        from trl import chat_template_utils
        pinned = json.loads((FIXTURES / "chat_templates" / "ghim_2026-09-26.json").read_text(encoding="utf-8"))["models"]
        for name, item in pinned.items():
            with self.subTest(name):
                model = find_model_config(ROOT / "configs" / "models" / "platform.json", name)
                self.assertEqual((model.source, model.revision), (item["source"], item["revision"]))
                template = getattr(chat_template_utils, item["trl_template"])
                self.assertEqual(hashlib.sha256(template.encode("utf-8")).hexdigest(), item["sha256"])  # template thật ở revision đã ghim = template TRL biết
                self.assertFalse(chat_template_utils.has_generation_markers(template))  # TRL sẽ thay bằng bản có {% generation %}


@unittest.skipIf(MISSING, f"thiếu thư viện: {', '.join(MISSING)}")
class AssistantOnlyLossTrainingTests(unittest.TestCase):
    ROWS = [{"messages": [{"role": "user", "content": f"Câu hỏi số {index}: thủ đô của Việt Nam là gì?"}, {"role": "assistant", "content": f"Trả lời số {index}: Hà Nội."}]} for index in range(6)]

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(); self.root = Path(self.directory.name)
        self.environ = mock.patch.dict(os.environ, helpers("test_m6_cpu_pipeline").OFFLINE); self.environ.start()
        (self.root / "sft.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in self.ROWS), encoding="utf-8")
        texts = [message["content"] for row in self.ROWS for message in row["messages"]] + ["You are Qwen, created by Alibaba Cloud. You are a helpful assistant."]
        helpers("test_m6_cpu_pipeline").make_tiny_model(self.root / "tiny", texts)
        entry = {"name": "tiny", "source": str(self.root / "tiny"), "dtype": "float32", "device_map": None, "offline": True, "kind": "text", "params_b": 0.0001, "capabilities": ["chat"]}
        (self.root / "models.json").write_text(json.dumps({"models": [entry]}), encoding="utf-8")

    def tearDown(self):
        self.environ.stop(); self.directory.cleanup()

    def use_template(self, template: str) -> None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.root / "tiny"); tokenizer.chat_template = template; tokenizer.save_pretrained(self.root / "tiny")

    def config(self, name: str, assistant_only_loss: bool) -> FinetuneConfig:
        return FinetuneConfig(base_model="tiny", models_config=str(self.root / "models.json"), dataset_path=str(self.root / "sft.jsonl"), output_dir=str(self.root / name),
                              method="lora", lora=LoraSettings(r=4, alpha=8, dropout=0.0), seed=0, max_steps=1, per_device_batch_size=2, gradient_accumulation_steps=1,
                              max_length=512, gradient_checkpointing=False, logging_steps=1, save_steps=1000, require_gpu=False, assistant_only_loss=assistant_only_loss)

    def train_and_capture(self, config: FinetuneConfig) -> tuple[dict, list]:
        """Train thật trên CPU; ghi lại input_ids và labels của từng batch mà SFTTrainer đưa vào hàm tính loss."""
        import trl
        from local_ai.training.finetune import train
        captured, original = [], trl.SFTTrainer.compute_loss
        def spy(trainer, model, inputs, *args, **kwargs):
            captured.append((inputs["input_ids"].detach().clone(), inputs["labels"].detach().clone())); return original(trainer, model, inputs, *args, **kwargs)
        with mock.patch.object(trl.SFTTrainer, "compute_loss", spy), contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            result = train(config)
        return result, captured

    def test_question_labels_are_masked_with_qwen_templates(self):
        from transformers import AutoTokenizer
        from trl import chat_template_utils
        for name in ("qwen2_5_chat_template", "qwen3_chat_template", "qwen3_8_chat_template"):  # template của smoke, light và model chính
            with self.subTest(name):
                self.use_template(getattr(chat_template_utils, name))
                tokenizer = AutoTokenizer.from_pretrained(self.root / "tiny")
                result, captured = self.train_and_capture(self.config(f"run-{name}", True))
                self.assertEqual((result["status"], result["steps"]), ("completed", 1))
                ids, labels = captured[0]
                for row in range(ids.shape[0]):
                    text, trained = tokenizer.decode(ids[row]), tokenizer.decode(ids[row][labels[row] != -100])
                    question = re.search(r"Câu hỏi số \d+: thủ đô của Việt Nam là gì\?", text).group(0)
                    answer = re.search(r"Trả lời số \d+: Hà Nội\.", text).group(0)
                    self.assertNotIn("Câu hỏi", trained); self.assertNotIn("user", trained)  # phần câu hỏi có nhãn -100
                    self.assertIn(answer, trained); self.assertIn("<|im_end|>", trained)  # câu trả lời và dấu kết thúc lượt vẫn được học
                    self.assertTrue(question and bool((labels[row][:5] == -100).all()))  # đầu chuỗi (lời hệ thống, câu hỏi) không được học

    def test_without_the_option_questions_are_trained_as_before(self):
        from transformers import AutoTokenizer
        from trl import chat_template_utils
        self.use_template(chat_template_utils.qwen2_5_chat_template)
        tokenizer = AutoTokenizer.from_pretrained(self.root / "tiny")
        _, captured = self.train_and_capture(self.config("run-cu", False))
        ids, labels = captured[0]
        self.assertIn("Câu hỏi số", tokenizer.decode(ids[0][labels[0] != -100]))  # hành vi cũ: tính loss cả câu hỏi

    def test_unsupported_template_fails_in_vietnamese_before_loading_the_model(self):
        from local_ai.training import finetune
        with mock.patch.object(finetune, "model_loader_class", side_effect=AssertionError("không được nạp model")), self.assertRaises(ValueError) as caught:
            finetune.train(self.config("run-loi", True))  # tokenizer tí hon của M6 dùng template tự viết, TRL không biết
        message = str(caught.exception)
        for phrase in ("chưa được TRL hỗ trợ", "chỉ tính loss trên câu trả lời", '"assistant_only_loss": false', "{% generation %}"):
            with self.subTest(phrase): self.assertIn(phrase, message)
        self.assertFalse((self.root / "run-loi").exists())  # chưa train gì
        result, _ = self.train_and_capture(self.config("run-tat", False))  # tắt bằng cấu hình thì train được
        self.assertEqual(result["status"], "completed")

    def test_check_accepts_templates_with_generation_markers(self):
        from transformers import AutoTokenizer
        self.use_template("{% for message in messages %}{{ message['role'] }}: {% if message['role'] == 'assistant' %}{% generation %}{{ message['content'] }}{% endgeneration %}{% else %}{{ message['content'] }}{% endif %}\n{% endfor %}")
        model = find_model_config(self.root / "models.json", "tiny")
        self.assertIn("đã có {% generation %}", check_assistant_only_loss(AutoTokenizer.from_pretrained(self.root / "tiny"), model))
        tokenizer = AutoTokenizer.from_pretrained(self.root / "tiny"); tokenizer.chat_template = None
        with self.assertRaisesRegex(ValueError, "không có chat template"): check_assistant_only_loss(tokenizer, model)


if __name__ == "__main__":
    unittest.main()
