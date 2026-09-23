"""Hugging Face dataset step: download rows, map them onto the dataset schema, then validate and export SFT data."""
from __future__ import annotations

import os
import re
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Iterable

from local_ai.data.core import build_dataset, write_jsonl

RowLoader = Callable[..., Iterable[dict[str, Any]]]


def hf_token() -> str | None:
    """HF_TOKEN is read only from the environment; public datasets work without it."""
    return os.environ.get("HF_TOKEN") or None


def validate_spec(spec: dict[str, Any]) -> None:
    if not spec.get("name"): raise ValueError("hf_dataset.name must name a Hugging Face dataset (org/name)")
    if not spec.get("license", {}).get("name"): raise ValueError("hf_dataset.license.name is required; check the dataset card before use")
    mapping = spec.get("mapping", {})
    if not spec.get("messages_field") and not (mapping.get("input") and mapping.get("expected_output")):
        raise ValueError("Set hf_dataset.messages_field or hf_dataset.mapping.input and mapping.expected_output")


def load_hf_rows(spec: dict[str, Any], loader: RowLoader | None = None) -> list[dict[str, Any]]:
    """Load rows via `datasets.load_dataset` (or an injected loader with the same keyword interface)."""
    if loader is None:
        try:
            from datasets import load_dataset
        except ImportError as error: raise RuntimeError("Install the optional `datasets` package to download Hugging Face datasets") from error
        loader = load_dataset
    rows = loader(spec["name"], spec.get("subset"), split=spec.get("split", "train"), revision=spec.get("revision", "main"), streaming=spec.get("streaming", False), token=hf_token())
    limit = spec.get("limit")
    return [dict(row) for row in (islice(rows, limit) if limit else rows)]


def _from_messages(messages: Any) -> tuple[str | None, str | None]:
    if not isinstance(messages, list): return None, None
    users = [m.get("content") for m in messages if isinstance(m, dict) and m.get("role") == "user"]
    assistants = [m.get("content") for m in messages if isinstance(m, dict) and m.get("role") == "assistant"]
    return (users[0] if users else None), (assistants[-1] if assistants else None)


def map_rows(rows: list[dict[str, Any]], spec: dict[str, Any], version: str) -> list[dict[str, Any]]:
    """Map arbitrary HF columns onto DatasetExample records; invalid rows are left for validation to reject."""
    validate_spec(spec)
    mapping = spec.get("mapping", {}); slug = re.sub(r"[^a-z0-9]+", "-", spec["name"].lower()).strip("-")
    source = {"name": spec["name"], "url": f"https://huggingface.co/datasets/{spec['name']}", "revision": spec.get("revision", "main")}
    records = []
    for index, row in enumerate(rows):
        if spec.get("messages_field"): prompt, answer = _from_messages(row.get(spec["messages_field"]))
        else: prompt, answer = row.get(mapping["input"]), row.get(mapping["expected_output"])
        identifier = row.get(mapping["id"]) if mapping.get("id") else None
        record = {"id": f"{slug}-{identifier if identifier not in (None, '') else index}", "domain": spec.get("domain", "reasoning"), "task": spec.get("task", "sft"), "input": prompt, "expected_output": answer, "source": source, "license": dict(spec["license"]), "dataset_version": version, "metadata": {"hf_split": spec.get("split", "train"), "hf_row": index}}
        for field in ("context", "reasoning"):
            if mapping.get(field) and row.get(mapping[field]) not in (None, ""): record[field] = row[mapping[field]]
        records.append(record)
    return records


def prepare_hf_sft(config: dict[str, Any], output_dir: str | Path | None = None, loader: RowLoader | None = None) -> dict[str, Any]:
    """Download -> map -> validate/deduplicate/export via build_dataset. Writes raw.jsonl, train.jsonl, sft.jsonl, rejected.jsonl, manifest.json."""
    spec = config["hf_dataset"]; validate_spec(spec)
    version = config.get("dataset_version", "hf-v1"); output = Path(output_dir or config.get("output_dir", "data/processed/hf_sft"))
    raw = output / "raw.jsonl"; write_jsonl(raw, map_rows(load_hf_rows(spec, loader), spec, version))
    build_config = {"seed": config.get("seed", 0), "formats": config.get("formats", ["sft"]), "hf_dataset": {key: value for key, value in spec.items() if key != "token"}}
    return build_dataset([raw], output, version, build_config, config.get("eval_sources", []))
