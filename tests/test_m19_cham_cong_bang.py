"""Mốc M19: chấm công bằng hơn.

Sửa các điểm chấm chưa công bằng thấy khi chạy thật trên Colab (M15):
1. tắt được chế độ suy nghĩ của Qwen3 khi chấm (`--no-thinking` → `enable_thinking=False` vào chat template);
2. Bước 9 (chấm sau) đẩy báo cáo lên Hub ở `eval/sau` và luôn chấm lại (`--no-reuse`);
3. bộ chấm có 16 câu tool_use (trước là 8);
4. TerminalTool từ chối lệnh thì kèm gợi ý; nhiệm vụ agent mới phải thử lại sau khi bị từ chối;
5. chấm lặp lại được: greedy cho model transformers, temperature 0 và seed cố định cho model qua server;
6. chấm nhiệm vụ agent so số theo giá trị;
7. ghim revision (mã commit) cho các model transformers.

Không cần mạng hay GPU: model tí hon khởi tạo ngẫu nhiên trên CPU, server OpenAI giả, huggingface_hub giả.
"""
import contextlib
import importlib.util
import io
import json
import os
import re
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from local_ai.agents.tasks import AgentTask, answer_matches, load_tasks, run_tasks
from local_ai.config.settings import find_model_config, load_model_configs
from local_ai.contracts import Message
from local_ai.evaluation.__main__ import eval_settings, main as evaluation_main
from local_ai.evaluation.suite import load_cases, markdown_report, score
from local_ai.models.adapters import HuggingFaceModelAdapter, ModelConfig
from local_ai.models.openai_compatible import OpenAICompatibleAdapter
from local_ai.models.router import ScriptedModelAdapter
from local_ai.runtime.terminal import CommandRejected, TerminalConfig, TerminalTool
from local_ai.training import calibrate
from local_ai.training.finetune import load_finetune_config

ROOT = Path(__file__).resolve().parent.parent
PLATFORM = ROOT / "configs" / "models" / "platform.json"
CASES = ROOT / "data" / "eval" / "eval_v1.jsonl"
LIBRARIES = ("torch", "transformers", "tokenizers")
MISSING = [name for name in LIBRARIES if importlib.util.find_spec(name) is None]
# Chat template kiểu Qwen3: enable_thinking=False thì chèn sẵn khối suy nghĩ rỗng, model trả lời luôn.
QWEN3_TEMPLATE = ("{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}"
                  "{% if add_generation_prompt %}<|im_start|>assistant\n{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}{% endif %}")


def helpers(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_helpers", Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def notebook_cells() -> dict[str, str]:
    notebook = json.loads((ROOT / "notebooks" / "train_colab.ipynb").read_text(encoding="utf-8"))
    return {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"}


class FakeInputs(dict):
    def to(self, device): return self


class FakeIds:
    shape = (1, 3)


class FakeProcessor:
    """Processor giả của model multimodal: ghi lại tham số truyền vào apply_chat_template."""
    def __init__(self): self.calls = []
    def apply_chat_template(self, conversation, **kwargs): self.calls.append(kwargs); return FakeInputs(input_ids=FakeIds())
    def decode(self, ids, skip_special_tokens): return "trả lời"


class FakeTokenizer(FakeProcessor):
    """Tokenizer giả của model chữ: apply_chat_template trả chuỗi prompt, gọi tokenizer thì trả input_ids."""
    def apply_chat_template(self, conversation, **kwargs): self.calls.append(kwargs); return "prompt"
    def __call__(self, prompt, return_tensors): return FakeInputs(input_ids=FakeIds())


class FakeModel:
    device = "cpu"
    def __init__(self): self.calls = []
    def generate(self, **kwargs): self.calls.append(kwargs); return [[0, 0, 0, 1]]


class ThinkingTests(unittest.TestCase):
    def test_no_thinking_flag_changes_settings_so_old_hub_reports_are_not_reused(self):
        base = find_model_config(PLATFORM, "light")
        thinking, no_thinking = eval_settings(base, CASES), eval_settings(replace(base, enable_thinking=False), CASES)
        self.assertIsNone(thinking["generation"]["enable_thinking"]); self.assertIs(no_thinking["generation"]["enable_thinking"], False)
        self.assertNotEqual(thinking, no_thinking)
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            self.assertEqual(evaluation_main(["--model", "light", "--no-thinking", "--dry-run"]), 0)
        self.assertIs(json.loads(printed.getvalue())["settings"]["generation"]["enable_thinking"], False)

    def test_report_with_thinking_on_hub_is_not_reused_with_no_thinking(self):
        m16 = helpers("test_m16_notebook_resilience")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "models.json"
            models.write_text(json.dumps({"models": [{"name": "may-chu-tat", "backend": "openai_compatible", "base_url": "http://127.0.0.1:9/v1", "source": "gia", "capabilities": ["chat"], "timeout_s": 2, "max_new_tokens": 64}]}), encoding="utf-8")
            hub = m16.FakeHub(root / "hub")
            cached = {"status": "completed", "model": "may-chu-tat", "generated_at": "2026-09-25T03:00:00+00:00", "cases": 38, "passed": 7, "accuracy": 0.18, "groups": {}, "languages": {}, "failures": [],
                      "results": [], "settings": eval_settings(find_model_config(models, "may-chu-tat"), CASES)}
            target = root / "hub" / "eval" / "truoc" / "report.json"; target.parent.mkdir(parents=True); target.write_text(json.dumps(cached), encoding="utf-8")
            argv = ["--model", "may-chu-tat", "--models", str(models), "--hub-repo", m16.REPO, "--hub-path", "eval/truoc"]
            with hub.patch(), contextlib.redirect_stdout(io.StringIO()) as printed, contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(evaluation_main([*argv, "--output", str(root / "cung")]), 0)  # cùng cài đặt: dùng lại
                self.assertEqual(evaluation_main([*argv, "--output", str(root / "khac"), "--no-thinking"]), 1)  # khác cài đặt: chấm lại (server tắt nên lỗi)
            self.assertIn("Dùng lại báo cáo", printed.getvalue()); self.assertIn("cách sinh chữ", printed.getvalue())

    def test_text_and_multimodal_adapters_pass_enable_thinking_to_chat_template(self):
        for kind in ("text", "multimodal"):
            with self.subTest(kind):
                adapter = HuggingFaceModelAdapter(ModelConfig(name="gia", source="gia", kind=kind, enable_thinking=False))
                processor, model = (FakeTokenizer() if kind == "text" else FakeProcessor()), FakeModel()
                adapter.model, adapter.tokenizer = model, processor
                self.assertEqual(adapter.generate([Message("user", "xin chào")]), "trả lời")
                self.assertIs(processor.calls[0]["enable_thinking"], False)
                self.assertEqual({key: model.calls[0][key] for key in ("do_sample", "max_new_tokens")}, {"do_sample": False, "max_new_tokens": 512})
        adapter = HuggingFaceModelAdapter(ModelConfig(name="gia", source="gia", kind="multimodal"))
        adapter.model, adapter.tokenizer = FakeModel(), FakeProcessor()
        adapter.generate([Message("user", "xin chào")])
        self.assertNotIn("enable_thinking", adapter.tokenizer.calls[0])  # không đặt thì để mặc định của chat template

    def test_notebook_uses_the_same_eval_settings_before_and_after_training(self):
        cells = notebook_cells()
        commands = [re.search(r'run\(f"(python -m local_ai\.evaluation [^"]+)"', cells[step]).group(1) for step in ("buoc-7-cham-truoc", "buoc-9-cham-sau")]
        options = [set(re.findall(r"--(?:max-new-tokens \{MAX_NEW_TOKENS\}|no-thinking)", command)) for command in commands]
        self.assertEqual(options[0], {"--max-new-tokens {MAX_NEW_TOKENS}", "--no-thinking"})
        self.assertEqual(options[0], options[1])

    def test_server_model_receives_enable_thinking(self):
        m13 = helpers("test_m13_agent_colab")
        with m13.FakeOpenAIServer(["ok"]) as server:
            config = ModelConfig(name="may-chu", source="qwen3:4b", backend="openai_compatible", base_url=server.base_url, enable_thinking=False)
            OpenAICompatibleAdapter(config).generate([Message("user", "xin chào")])
        body = server.requests[0]["body"]
        self.assertEqual((body["temperature"], body["seed"], body["chat_template_kwargs"]), (0.0, 0, {"enable_thinking": False}))


class AfterTrainingReportTests(unittest.TestCase):
    def test_step_9_pushes_to_eval_sau_and_always_evaluates(self):
        cell = notebook_cells()["buoc-9-cham-sau"]
        self.assertIn("--hub-repo {HUB_REPO} --hub-path eval/sau --no-reuse", cell)
        self.assertIn("--hub-path eval/truoc", notebook_cells()["buoc-7-cham-truoc"]); self.assertNotIn("--no-reuse", notebook_cells()["buoc-7-cham-truoc"])

    def test_no_reuse_evaluates_again_and_pushes_the_new_report(self):
        m16 = helpers("test_m16_notebook_resilience")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); hub = m16.FakeHub(root / "hub")
            argv = ["--scripted", "--hub-repo", m16.REPO, "--hub-path", "eval/sau"]
            with hub.patch(), contextlib.redirect_stdout(io.StringIO()) as printed:
                self.assertEqual(evaluation_main([*argv, "--output", str(root / "lan1")]), 0)
                self.assertEqual(evaluation_main([*argv, "--output", str(root / "lan2"), "--no-reuse"]), 0)
            self.assertNotIn("Dùng lại", printed.getvalue())
            self.assertEqual([call for call in hub.calls if call[0] != "create_repo"], [("download", "eval/sau/report.json"), ("upload", "eval/sau/report.json"), ("upload", "eval/sau/report.md"),
                                                                                    ("upload", "eval/sau/report.json"), ("upload", "eval/sau/report.md")])  # lần 2 không tải về, chấm lại rồi đẩy lên
            self.assertEqual(json.loads((root / "hub" / "eval" / "sau" / "report.json").read_text(encoding="utf-8"))["generated_at"],
                             json.loads((root / "lan2" / "report.json").read_text(encoding="utf-8"))["generated_at"])
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            self.assertEqual(evaluation_main(["--scripted", "--dry-run", "--hub-repo", m16.REPO, "--hub-path", "eval/sau", "--no-reuse"]), 0)
        self.assertIs(json.loads(printed.getvalue())["reuse_hub_report"], False)


class ToolUseCasesTests(unittest.TestCase):
    def test_at_least_16_tool_use_cases_in_both_languages(self):
        tools = [case for case in load_cases() if case.group == "tool_use"]
        self.assertGreaterEqual(len(tools), 16)
        self.assertEqual(sum(case.language == "vi" for case in tools), sum(case.language == "en" for case in tools))
        self.assertEqual(len({case.prompt for case in tools}), len(tools))
        for name in ("calculator", "read_file", "search", None):  # mỗi ngôn ngữ có đủ 3 công cụ và câu không cần công cụ
            for language in ("vi", "en"):
                with self.subTest(f"{name}/{language}"):
                    self.assertGreaterEqual(sum(json.loads(case.reference)["tool"] == name and case.language == language for case in tools), 2)

    def test_each_tool_case_rejects_the_other_answers(self):
        tools = [case for case in load_cases() if case.group == "tool_use"]
        for case in tools:
            self.assertTrue(score(case, case.reference)[0], case.id)
            for other in tools:
                if json.loads(other.reference) != json.loads(case.reference):
                    with self.subTest(f"{case.id} <- {other.id}"): self.assertFalse(score(case, other.reference)[0])

    def test_eval_time_estimate_uses_each_report_case_count(self):
        m15 = helpers("test_m15_measurements")
        real = m15.fixture("that_light_t4_2026-09-25.json")
        report = dict(real["eval_before"])
        result = calibrate.compare(m15.config_without_data("light"), real["measurements"], {"before": report, "after": {**report, "cases": 38}}, max_new_tokens=512)
        before, after = result["eval"]["before"]["estimated_max_minutes"], result["eval"]["after"]["estimated_max_minutes"]
        self.assertEqual(result["eval"]["before"]["cases"], 30)
        self.assertAlmostEqual(after / before, 38 / 30, delta=0.01)


class TerminalHintTests(unittest.TestCase):
    def tool(self, directory: str, enabled: bool = True) -> TerminalTool:
        return TerminalTool(replace(TerminalConfig.from_file(), workspace=Path(directory), log_path=Path(directory) / "log.jsonl"), enabled=enabled)

    def test_rejections_suggest_allowed_commands_one_at_a_time(self):
        with tempfile.TemporaryDirectory() as directory:
            tool = self.tool(directory)
            for command in ("wc -l du_lieu.csv", "cat du_lieu.csv | wc -l", "ls > danh_sach.txt", "cat /etc/passwd", "git push"):
                with self.subTest(command), self.assertRaises(CommandRejected) as caught: tool(command)
                message = str(caught.exception)
                for phrase in ("Gợi ý:", "từng lệnh một", "pipe (|)", "redirect (> <)", "cat <tên file>", "Hint: allowed commands are cat, date, find, git", "Run one command at a time"):
                    with self.subTest(f"{command}: {phrase}"): self.assertIn(phrase, message)
            with self.assertRaises(CommandRejected) as caught: self.tool(directory, enabled=False)("ls")
            self.assertNotIn("Gợi ý", str(caught.exception))  # công cụ tắt: không có lệnh nào thay thế được
            logged = [json.loads(line) for line in (Path(directory) / "log.jsonl").read_text(encoding="utf-8").splitlines()]
            rejected = [entry["reason"] for entry in logged if not entry["allowed"] and "đang tắt" not in entry["reason"]]
            self.assertEqual(len(rejected), 5); self.assertTrue(all("Gợi ý" in reason for reason in rejected))  # log ghi cả gợi ý

    def test_retry_task_passes_only_after_a_rejection_and_a_successful_retry(self):
        task = next(task for task in load_tasks() if task.retry_after_rejection)
        self.assertEqual((task.id, task.tools, task.expected), ("thu-lai-khi-bi-tu-choi", ("terminal",), ("3",)))
        self.assertIn("wc -l", task.prompt)
        skip = AgentTask("khong-thu", task.prompt, task.expected, task.tools, ("plan", '{"tool": "terminal", "arguments": {"command": "cat du_lieu.csv"}}', '{"complete": true, "answer": "Có 3 dòng dữ liệu."}'), True)
        give_up = AgentTask("bo-cuoc", task.prompt, task.expected, task.tools, ("plan", '{"tool": "terminal", "arguments": {"command": "wc -l du_lieu.csv"}}', '{"complete": true, "answer": "3"}',
                                                                               '{"tool": "calculator", "arguments": {"expression": "4 - 1"}}', '{"complete": true, "answer": "Có 3 dòng."}'), True)
        with tempfile.TemporaryDirectory() as directory:
            report = run_tasks([task, skip, give_up], lambda item: ScriptedModelAdapter("mau", list(item.reference)), directory)
        good, skipped, gave_up = report["tasks"]
        self.assertEqual((good["success"], good["rejected_commands"], good["tools_used"]), (True, 1, ["terminal"]))
        self.assertTrue(any("Hint: allowed commands" in step for step in good["trace"]))
        self.assertEqual((skipped["success"], skipped["reason"]), (False, "không có lệnh bị TerminalTool từ chối rồi thử lại thành công"))
        self.assertEqual((gave_up["success"], gave_up["reason"]), (False, "không gọi công cụ bắt buộc: terminal"))

    def test_retry_task_through_fake_openai_server_sends_hint_to_model(self):
        m13 = helpers("test_m13_agent_colab")
        task = next(task for task in load_tasks() if task.retry_after_rejection)
        with m13.FakeOpenAIServer(task.reference) as server, tempfile.TemporaryDirectory() as directory:
            model = OpenAICompatibleAdapter(ModelConfig.from_dict({"name": "ollama-gia", "source": "qwen3:4b", "backend": "openai_compatible", "base_url": server.base_url, "capabilities": ["chat", "reasoning", "tool_calling"]}))
            report = run_tasks([task], lambda item: model, directory)
        self.assertTrue(report["tasks"][0]["success"], report["tasks"][0]["reason"])
        retry = json.loads(server.requests[3]["body"]["messages"][-1]["content"])  # lần hỏi model sau khi bị từ chối
        self.assertIn("Hint: allowed commands", retry["observation"])
        self.assertTrue(all(request["body"]["temperature"] == 0.0 and request["body"]["seed"] == 0 for request in server.requests))


@unittest.skipIf(MISSING, f"thiếu thư viện: {', '.join(MISSING)}")
class ReproducibleEvalTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(); self.root = Path(self.directory.name)
        self.environ = mock.patch.dict(os.environ, helpers("test_m6_cpu_pipeline").OFFLINE); self.environ.start()
        from transformers import AutoTokenizer, GenerationConfig
        cases = load_cases()
        helpers("test_m6_cpu_pipeline").make_tiny_model(self.root / "tiny", [case.prompt for case in cases])
        tokenizer = AutoTokenizer.from_pretrained(self.root / "tiny"); tokenizer.chat_template = QWEN3_TEMPLATE; tokenizer.save_pretrained(self.root / "tiny")
        GenerationConfig(do_sample=True, temperature=0.7, top_p=0.8, top_k=20).save_pretrained(self.root / "tiny")  # như generation_config.json của Qwen: lấy mẫu ngẫu nhiên
        self.config = ModelConfig(name="tiny", source=str(self.root / "tiny"), revision="main", dtype="float32", device_map=None, offline=True, max_new_tokens=8)
        self.cases = [case for case in cases if case.group == "tool_use"][:4] + [case for case in cases if case.group == "reasoning"][:2]

    def tearDown(self):
        self.environ.stop(); self.directory.cleanup()

    def evaluate(self, config: ModelConfig) -> dict:
        from local_ai.evaluation.suite import run_eval
        with contextlib.redirect_stderr(io.StringIO()):
            return run_eval(HuggingFaceModelAdapter(config), self.cases)  # adapter mới mỗi lần: nạp lại model từ đầu

    def test_default_generation_config_samples_randomly(self):
        import torch
        adapter = HuggingFaceModelAdapter(self.config); adapter.load()
        inputs = adapter.tokenizer("xin chào", return_tensors="pt")
        outputs = []
        for seed in (1, 2):
            torch.manual_seed(seed); outputs.append(adapter.model.generate(**inputs, max_new_tokens=8)[0].tolist())
        self.assertNotEqual(outputs[0], outputs[1])  # lỗi cũ: generate() không truyền gì thêm thì lấy mẫu theo cấu hình của model

    def test_two_evaluations_give_identical_results(self):
        first, second = self.evaluate(self.config), self.evaluate(self.config)
        self.assertEqual(first["status"], "completed")
        self.assertEqual([item["output"] for item in first["results"]], [item["output"] for item in second["results"]])
        self.assertEqual((first["passed"], first["failures"]), (second["passed"], second["failures"]))
        sampled = replace(self.config, temperature=0.9, seed=7)
        self.assertEqual([item["output"] for item in self.evaluate(sampled)["results"]], [item["output"] for item in self.evaluate(sampled)["results"]])  # lấy mẫu có seed cũng lặp lại được

    def test_no_thinking_reaches_the_chat_template(self):
        adapter = HuggingFaceModelAdapter(replace(self.config, enable_thinking=False)); adapter.load()
        prompts, original = [], adapter.tokenizer.apply_chat_template
        adapter.tokenizer.apply_chat_template = lambda *args, **kwargs: prompts.append(original(*args, **kwargs)) or prompts[-1]
        adapter.generate([Message("user", "xin chào")])
        self.assertTrue(prompts[0].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), prompts[0])

    def test_settings_and_report_record_generation_and_revision(self):
        models = self.root / "models.json"
        models.write_text(json.dumps({"models": [{"name": "tiny", "source": str(self.root / "tiny"), "revision": "main", "dtype": "float32", "device_map": None, "offline": True, "kind": "text",
                                                  "capabilities": ["chat"], "max_new_tokens": 8}]}), encoding="utf-8")
        cases = self.root / "cases.jsonl"
        cases.write_text("".join(json.dumps({key: value for key, value in vars(case).items() if value is not None}, ensure_ascii=False) + "\n" for case in self.cases), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            codes = [evaluation_main(["--model", "tiny", "--models", str(models), "--cases", str(cases), "--no-thinking", "--output", str(self.root / name)]) for name in ("lan1", "lan2")]
        reports = [json.loads((self.root / name / "report.json").read_text(encoding="utf-8")) for name in ("lan1", "lan2")]
        self.assertEqual((codes, [report["status"] for report in reports]), ([0, 0], ["completed", "completed"]))
        self.assertEqual(reports[0]["settings"], reports[1]["settings"])
        self.assertEqual([item["output"] for item in reports[0]["results"]], [item["output"] for item in reports[1]["results"]])
        self.assertEqual(reports[0]["settings"]["generation"], {"do_sample": False, "temperature": 0.0, "seed": 0, "enable_thinking": False})
        self.assertEqual(reports[0]["settings"]["revision"], "main")
        markdown = (self.root / "lan1" / "report.md").read_text(encoding="utf-8")
        for phrase in ("Cài đặt:", "revision `main`", "greedy (do_sample=False)", "tắt suy nghĩ (enable_thinking=False)", "tối đa 8 token"):
            with self.subTest(phrase): self.assertIn(phrase, markdown)


class GenerationConfigTests(unittest.TestCase):
    def test_defaults_are_greedy_and_invalid_values_are_rejected(self):
        config = find_model_config(PLATFORM, "smoke")
        self.assertEqual(config.generation_settings, {"do_sample": False, "temperature": 0.0, "seed": 0, "enable_thinking": None})
        for bad in ({"temperature": -0.1}, {"temperature": True}, {"seed": -1}, {"seed": 1.5}, {"enable_thinking": "false"}):
            with self.subTest(bad), self.assertRaises(ValueError): ModelConfig(name="x", source="x", **bad)

    def test_markdown_report_describes_sampling(self):
        report = {"status": "completed", "model": "m", "generated_at": "t", "cases": 1, "passed": 1, "accuracy": 1.0, "groups": {}, "languages": {}, "failures": [],
                  "settings": {"model": "m", "source": "s", "revision": "abc", "max_new_tokens": 16, "generation": {"do_sample": True, "temperature": 0.7, "seed": 3, "enable_thinking": None}, "cases_sha256": "0" * 64}}
        text = markdown_report(report)
        self.assertIn("lấy mẫu temperature 0.7, seed 3", text); self.assertIn("suy nghĩ theo mặc định của chat template", text)


class NumericAnswerTests(unittest.TestCase):
    def test_numbers_are_compared_by_value(self):
        for answer, expected, matches in (("Tổng là 187.", "87", False), ("Tổng là 87,5.", "87", False), ("87.5", "87", False), ("1870", "87", False), ("Có 13 dòng", "3", False),
                                          ("Tổng cột so_luong là 87.", "87", True), ("Khách trả 337500.0 đồng", "337500", True), ("Khách phải trả 337.500 đồng.", "337500", True),
                                          ("337 500 đ", "337500", True), ("(17 * 23) + 158 = 549", "549", True), ("tổng 87đ", "87", True), ("-5 độ", "-5", True), ("5 độ", "-5", False)):
            with self.subTest(f"{answer} ~ {expected}"): self.assertIs(answer_matches(answer, (expected,)), matches)

    def test_text_items_still_use_substring(self):
        self.assertTrue(answer_matches("mã số là dh-4827", ("DH-4827",)))
        self.assertFalse(answer_matches("Có 2 file: bao_cao.md, du_lieu.csv", ("bao_cao.md", "du_lieu.csv", "ghi_chu.txt")))
        self.assertTrue(answer_matches("Tổng 87 từ file du_lieu.csv", ("87", "du_lieu.csv")))


class PinnedRevisionTests(unittest.TestCase):
    def test_transformers_models_pin_commit_and_variants_share_it(self):
        configs = {config.name: config for config in load_model_configs(PLATFORM)}
        bases = {config.source: config.revision for name, config in configs.items() if config.backend == "transformers" and not config.adapter_path and not name.endswith(("-lora", "-colab"))}
        self.assertEqual(set(bases), {"huihui-ai/Huihui-Qwen3.8-27B-abliterated", "Qwen/Qwen2.5-Coder-7B-Instruct", "Qwen/Qwen3-4B", "Qwen/Qwen2.5-0.5B-Instruct"})
        for name, config in configs.items():
            if config.backend != "transformers": continue
            with self.subTest(name):
                self.assertRegex(config.revision, r"^[0-9a-f]{40}$")
                self.assertEqual(config.revision, bases[config.source])  # -lora, -colab dùng cùng revision với model gốc
                if config.tokenizer_revision: self.assertEqual(config.tokenizer_revision, config.revision)

    def test_training_and_eval_use_the_pinned_revision(self):
        smoke = find_model_config(PLATFORM, "smoke")
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            evaluation_main(["--model", "smoke-colab", "--dry-run"])
        self.assertEqual(json.loads(printed.getvalue())["settings"]["revision"], smoke.revision)
        from local_ai.training.finetune import describe
        self.assertEqual(describe(load_finetune_config(ROOT / "configs" / "training" / "colab_smoke.json"))["base_model"]["revision"], smoke.revision)


if __name__ == "__main__":
    unittest.main()
