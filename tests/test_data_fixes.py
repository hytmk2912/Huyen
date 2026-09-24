"""Kiểm tra 3 lỗi làm hỏng dữ liệu train: hội thoại nhiều lượt, mất reasoning/context, rò rỉ eval."""

import tempfile
import unittest
from pathlib import Path

from local_ai.data.core import build_dataset, load_records, prepare_format, validate_records, write_jsonl
from local_ai.data.hub import map_rows

LICENSE = {"name": "CC0-1.0"}
SPEC = {"name": "test-org/chat", "license": LICENSE, "messages_field": "messages", "mapping": {}}
CONVERSATION = [
    {"role": "system", "content": "Bạn là trợ lý địa lý."},
    {"role": "user", "content": "Thủ đô Pháp?"},
    {"role": "assistant", "content": "Paris"},
    {"role": "user", "content": "Còn Đức?"},
    {"role": "assistant", "content": "Berlin"},
]


def record(identifier, question, answer="đáp án", **extra):
    return {
        "id": identifier,
        "domain": "reasoning",
        "task": "t",
        "input": question,
        "expected_output": answer,
        "source": {"name": "test"},
        "license": LICENSE,
        "dataset_version": "v1",
        **extra,
    }


class MultiTurnTests(unittest.TestCase):
    def test_multi_turn_conversation_is_kept_whole(self):
        mapped = map_rows([{"messages": CONVERSATION}], SPEC, "v1")
        valid, rejected = validate_records(mapped)
        self.assertEqual((len(valid), len(rejected)), (1, 0))
        self.assertEqual(prepare_format(valid, "sft")[0]["messages"], CONVERSATION)

    def test_broken_conversations_are_rejected(self):
        broken = [
            [{"role": "user", "content": "Hỏi"}],
            [{"role": "robot", "content": "x"}, {"role": "assistant", "content": "y"}],
            "không phải danh sách",
        ]
        valid, rejected = validate_records(map_rows([{"messages": item} for item in broken], SPEC, "v1"))
        self.assertEqual((len(valid), len(rejected)), (0, 3))


class ReasoningContextTests(unittest.TestCase):
    def test_sft_keeps_reasoning_and_context(self):
        sft = prepare_format(
            [record("a", "2 + 3 = ?", "5", reasoning="Cộng 2 với 3.", context="Phép cộng số tự nhiên.")], "sft"
        )[0]["messages"]
        self.assertEqual(sft[0], {"role": "user", "content": "Phép cộng số tự nhiên.\n\n2 + 3 = ?"})
        self.assertEqual(sft[1], {"role": "assistant", "content": "<think>\nCộng 2 với 3.\n</think>\n\n5"})

    def test_plain_record_is_unchanged(self):
        self.assertEqual(
            prepare_format([record("a", "Hỏi", "Đáp")], "sft")[0]["messages"],
            [{"role": "user", "content": "Hỏi"}, {"role": "assistant", "content": "Đáp"}],
        )


class EvalLeakTests(unittest.TestCase):
    def build(self, train, evaluation):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(root / "train.jsonl", train)
            write_jsonl(root / "eval.jsonl", evaluation)
            manifest = build_dataset(
                [root / "train.jsonl"], root / "out", "v1", {"formats": ["sft"]}, [root / "eval.jsonl"]
            )
            return manifest, load_records(root / "out" / "sft.jsonl"), load_records(root / "out" / "rejected.jsonl")

    def test_same_question_with_different_id_is_removed_not_fatal(self):
        manifest, sft, rejected = self.build(
            [record("hf-0", "Thủ đô của Việt Nam?"), record("hf-1", "Câu khác")],
            [record("eval-9", "  thủ đô của việt nam? ", "Hà Nội")],
        )
        self.assertEqual([item["id"] for item in sft], ["hf-1"])
        self.assertEqual(manifest["statistics"]["eval_overlap_removed"], 1)
        self.assertEqual(rejected[0]["verification"]["eval_overlap"], "question")

    def test_identical_content_and_eval_prompt_format(self):
        _, sft, rejected = self.build(
            [record("hf-0", "Câu A"), record("hf-1", "Câu B")],
            [record("eval-0", "Câu A"), {"id": "vi-1", "suite": "vietnamese", "prompt": "Câu B", "expected": "x"}],
        )
        self.assertEqual(sft, [])
        self.assertEqual({item["verification"]["eval_overlap"] for item in rejected}, {"content_hash", "question"})

    def test_multi_turn_later_question_is_checked(self):
        conversation = map_rows([{"messages": CONVERSATION}], SPEC, "v1")
        _, sft, _ = self.build(conversation, [record("eval-0", "Còn Đức?")])
        self.assertEqual(sft, [])


if __name__ == "__main__":
    unittest.main()
