import json
import tempfile
import unittest
from pathlib import Path

from local_ai.config.settings import find_model_config, load_model_configs
from local_ai.data.corpus import DOMAINS, Registry, Source, build_one, check_source_domains, domain_mixture, domain_targets
from local_ai.models.gguf import export_command, export_gguf

PLATFORM = "configs/models/platform.json"
CORPUS = "configs/datasets/corpus_10t.json"


class HuihuiQwen3ModelTests(unittest.TestCase):
    def test_three_huihui_qwen3_models_are_fp32_gguf(self):
        configs = load_model_configs(PLATFORM)
        self.assertEqual(configs[0].name, "primary")
        extra = configs[1:]
        self.assertEqual([config.name for config in extra], ["huihui-qwen3-4b", "huihui-qwen3-8b", "huihui-qwen3-14b"])
        for config in extra:
            self.assertTrue(config.source.startswith("huihui-ai/Huihui-Qwen3-"))
            self.assertEqual(config.dtype, "float32")
            self.assertEqual(config.gguf_outtype, "f32")
            self.assertTrue(config.gguf_file.endswith(".f32.gguf"))

    def test_gguf_export_command_uses_llama_cpp_f32(self):
        config = find_model_config(PLATFORM, "huihui-qwen3-8b")
        command = export_command(config, "/snapshot", "/opt/llama.cpp")
        self.assertEqual(command[1:], ["/opt/llama.cpp/convert_hf_to_gguf.py", "/snapshot", "--outtype", "f32", "--outfile", config.gguf_file])
        plan = export_gguf(config, "/opt/llama.cpp", dry_run=True)
        self.assertEqual(plan["outtype"], "f32")

    def test_gguf_requires_configured_file(self):
        with self.assertRaisesRegex(ValueError, "gguf_file"): export_command(find_model_config(PLATFORM, "primary"), "/s", "/l")


class CorpusMixtureTests(unittest.TestCase):
    def test_10t_mixture_ratios_and_targets(self):
        config = json.loads(Path(CORPUS).read_text())
        self.assertEqual(domain_mixture(config), {"code": 0.10, "trading": 0.20, "reasoning": 0.10, "vietnamese": 0.10, "general": 0.50})
        self.assertEqual(domain_mixture({}), DOMAINS)
        targets = domain_targets(config)
        self.assertEqual(targets["trading"], 2_000_000_000_000)
        self.assertEqual(targets["general"], 5_000_000_000_000)
        self.assertEqual(sum(targets.values()), config["target_tokens"])

    def test_invalid_mixture_and_unknown_source_domain_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "1.0"): domain_mixture({"domain_mixture": {"code": 0.5, "general": 0.4}})
        config = json.loads(Path(CORPUS).read_text())
        sources = [Source.from_dict(item) for item in config["sources"]]
        check_source_domains(sources, config)
        stray = Source.from_dict({**config["sources"][0], "domain": "long_form_technical"})
        with self.assertRaisesRegex(ValueError, "long_form_technical"): check_source_domains([stray], config)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "domain_mixture"): build_one(stray, {**config, "storage_root": directory}, dry_run=True)

    def test_progress_reports_per_domain_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = Registry(Path(directory))
            registry.shard(("s", "v", "local", "checksum", 30, 1, "trading", "COMPLETE", "remote", "now"))
            progress = registry.progress(1000, DOMAINS)
        self.assertEqual(progress["by_domain_target"]["trading"]["target_tokens"], 200)
        self.assertEqual(progress["by_domain_target"]["trading"]["remaining_tokens"], 170)
        self.assertEqual(progress["by_domain_target"]["vietnamese"]["verified_tokens"], 0)


if __name__ == "__main__": unittest.main()
