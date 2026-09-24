"""Bước dữ liệu Hugging Face: tải các dòng, chuyển sang schema dữ liệu của repo, rồi kiểm tra và xuất dữ liệu SFT."""

from __future__ import annotations

import os
import re
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Iterable

from local_ai.data.core import build_dataset, write_jsonl

RowLoader = Callable[..., Iterable[dict[str, Any]]]


def hf_token() -> str | None:
    """HF_TOKEN chỉ được đọc từ biến môi trường; dataset công khai thì không cần."""
    return os.environ.get("HF_TOKEN") or None


def validate_spec(spec: dict[str, Any]) -> None:
    if not spec.get("name"):
        raise ValueError("hf_dataset.name phải là tên một dataset trên Hugging Face (dạng org/name)")
    if not spec.get("license", {}).get("name"):
        raise ValueError(
            "Bắt buộc có hf_dataset.license.name; hãy xem trang giới thiệu dataset (dataset card) trước khi dùng"
        )
    mapping = spec.get("mapping", {})
    if not spec.get("messages_field") and not (mapping.get("input") and mapping.get("expected_output")):
        raise ValueError("Hãy đặt hf_dataset.messages_field, hoặc hf_dataset.mapping.input và mapping.expected_output")


def load_hf_rows(spec: dict[str, Any], loader: RowLoader | None = None) -> list[dict[str, Any]]:
    """Tải các dòng bằng `datasets.load_dataset` (hoặc một hàm tải được truyền vào, có cùng tham số)."""
    if loader is None:
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise RuntimeError("Hãy cài gói tùy chọn `datasets` để tải dataset từ Hugging Face") from error
        loader = load_dataset
    rows = loader(
        spec["name"],
        spec.get("subset"),
        split=spec.get("split", "train"),
        revision=spec.get("revision", "main"),
        streaming=spec.get("streaming", False),
        token=hf_token(),
    )
    limit = spec.get("limit")
    return [dict(row) for row in (islice(rows, limit) if limit else rows)]


def _clean_messages(messages: Any) -> list[dict[str, str]] | None:
    """Giữ nguyên toàn bộ hội thoại (system + mọi lượt), chỉ lấy role/content; dữ liệu sai dạng để bước kiểm tra loại bỏ."""
    if not isinstance(messages, list):
        return None
    return [
        {"role": m.get("role"), "content": m.get("content")} if isinstance(m, dict) else {"role": None, "content": None}
        for m in messages
    ]


def map_rows(rows: list[dict[str, Any]], spec: dict[str, Any], version: str) -> list[dict[str, Any]]:
    """Chuyển các cột HF bất kỳ thành bản ghi DatasetExample; dòng không hợp lệ để bước kiểm tra loại bỏ."""
    validate_spec(spec)
    mapping = spec.get("mapping", {})
    slug = re.sub(r"[^a-z0-9]+", "-", spec["name"].lower()).strip("-")
    source = {
        "name": spec["name"],
        "url": f"https://huggingface.co/datasets/{spec['name']}",
        "revision": spec.get("revision", "main"),
    }
    records = []
    for index, row in enumerate(rows):
        messages = _clean_messages(row.get(spec["messages_field"])) if spec.get("messages_field") else None
        if spec.get("messages_field"):
            # input/expected_output chỉ để kiểm tra, loại trùng và chặn rò rỉ eval; sft.jsonl dùng nguyên messages.
            users = [m["content"] for m in messages or [] if m["role"] == "user"]
            prompt, answer = (users[0] if users else None), (messages[-1]["content"] if messages else None)
        else:
            prompt, answer = row.get(mapping["input"]), row.get(mapping["expected_output"])
        identifier = row.get(mapping["id"]) if mapping.get("id") else None
        record = {
            "id": f"{slug}-{identifier if identifier not in (None, '') else index}",
            "domain": spec.get("domain", "reasoning"),
            "task": spec.get("task", "sft"),
            "input": prompt,
            "expected_output": answer,
            "source": source,
            "license": dict(spec["license"]),
            "dataset_version": version,
            "metadata": {"hf_split": spec.get("split", "train"), "hf_row": index},
        }
        if messages is not None:
            record["messages"] = messages
        for field in ("context", "reasoning"):
            if mapping.get(field) and row.get(mapping[field]) not in (None, ""):
                record[field] = row[mapping[field]]
        records.append(record)
    return records


def prepare_hf_sft(
    config: dict[str, Any], output_dir: str | Path | None = None, loader: RowLoader | None = None
) -> dict[str, Any]:
    """Tải -> chuyển đổi -> kiểm tra/loại trùng/xuất qua build_dataset. Ghi ra raw.jsonl, train.jsonl, sft.jsonl, rejected.jsonl, manifest.json."""
    spec = config["hf_dataset"]
    validate_spec(spec)
    version = config.get("dataset_version", "hf-v1")
    output = Path(output_dir or config.get("output_dir", "data/processed/hf_sft"))
    raw = output / "raw.jsonl"
    write_jsonl(raw, map_rows(load_hf_rows(spec, loader), spec, version))
    build_config = {
        "seed": config.get("seed", 0),
        "formats": config.get("formats", ["sft"]),
        "hf_dataset": {key: value for key, value in spec.items() if key != "token"},
    }
    return build_dataset([raw], output, version, build_config, config.get("eval_sources", []))
