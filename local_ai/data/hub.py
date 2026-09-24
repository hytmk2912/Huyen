"""Bước dữ liệu Hugging Face: tải các dòng, chuyển sang schema dữ liệu của repo, rồi kiểm tra và xuất dữ liệu SFT.

Preset dataset nằm trong `configs/datasets/presets/<tên>.json`; có thể trộn nhiều preset theo tỉ lệ thành một sft.jsonl.
"""
from __future__ import annotations

import json
import math
import os
import re
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Iterable

from local_ai.data.core import build_dataset, write_jsonl

RowLoader = Callable[..., Iterable[dict[str, Any]]]
PRESET_DIR = Path(__file__).resolve().parents[2] / "configs" / "datasets" / "presets"
DEFAULT_SFT_DIR = "data/processed/hf_sft"  # nơi các cấu hình train (configs/training/*.json) đọc sft.jsonl
ROLE_ALIASES = {"human": "user", "gpt": "assistant"}  # dạng ShareGPT: {"from": "human", "value": "..."}
_state = {"real_stream": False}


def real_stream_used() -> bool:
    """True nếu tiến trình này đã đọc stream thật bằng thư viện datasets (xem ghi chú ở local_ai/data/__main__.py)."""
    return _state["real_stream"]


def hf_token() -> str | None:
    """HF_TOKEN chỉ được đọc từ biến môi trường; dataset công khai thì không cần."""
    return os.environ.get("HF_TOKEN") or None


def validate_spec(spec: dict[str, Any]) -> None:
    if not spec.get("name"): raise ValueError("hf_dataset.name phải là tên một dataset trên Hugging Face (dạng org/name)")
    if not spec.get("license", {}).get("name"): raise ValueError("Bắt buộc có hf_dataset.license.name; hãy xem trang giới thiệu dataset (dataset card) trước khi dùng")
    mapping = spec.get("mapping", {})
    if not spec.get("messages_field") and not (mapping.get("input") and mapping.get("expected_output")):
        raise ValueError("Hãy đặt hf_dataset.messages_field, hoặc hf_dataset.mapping.input và mapping.expected_output")


def load_hf_rows(spec: dict[str, Any], loader: RowLoader | None = None) -> list[dict[str, Any]]:
    """Tải các dòng bằng `datasets.load_dataset` (hoặc một hàm tải được truyền vào, có cùng tham số)."""
    if loader is None:
        try:
            from datasets import load_dataset
        except ImportError as error: raise RuntimeError("Hãy cài gói tùy chọn `datasets` để tải dataset từ Hugging Face") from error
        loader = load_dataset
        _state["real_stream"] = _state["real_stream"] or bool(spec.get("streaming"))
    rows = loader(spec["name"], spec.get("subset"), split=spec.get("split", "train"), revision=spec.get("revision", "main"), streaming=spec.get("streaming", False), token=hf_token())
    limit = spec.get("limit")
    return [dict(row) for row in (islice(rows, limit) if limit else rows)]


def clean_messages(messages: Any) -> list[dict[str, Any]] | None:
    """Giữ nguyên toàn bộ hội thoại (system và mọi lượt), chỉ lấy role/content; hiểu cả dạng ShareGPT (from/value).
    Lượt sai dạng vẫn giữ để bước kiểm tra loại cả bản ghi."""
    if not isinstance(messages, list): return None
    result = []
    for message in messages:
        if not isinstance(message, dict): result.append({"role": None, "content": None}); continue
        role = message.get("role", message.get("from"))
        result.append({"role": ROLE_ALIASES.get(role, role), "content": message.get("content", message.get("value"))})
    return result


def map_rows(rows: list[dict[str, Any]], spec: dict[str, Any], version: str) -> list[dict[str, Any]]:
    """Chuyển các cột HF bất kỳ thành bản ghi DatasetExample; dòng không hợp lệ để bước kiểm tra loại bỏ."""
    validate_spec(spec)
    mapping = spec.get("mapping", {}); slug = re.sub(r"[^a-z0-9]+", "-", spec["name"].lower()).strip("-")
    source = {"name": spec["name"], "url": f"https://huggingface.co/datasets/{spec['name']}", "revision": spec.get("revision", "main")}
    records = []
    for index, row in enumerate(rows):
        messages = clean_messages(row.get(spec["messages_field"])) if spec.get("messages_field") else None
        if spec.get("messages_field"):
            # input/expected_output chỉ dùng để kiểm tra và loại trùng; sft.jsonl dùng nguyên messages.
            users = [m["content"] for m in messages or [] if m["role"] == "user"]
            prompt, answer = (users[0] if users else None), (messages[-1]["content"] if messages else None)
        else: prompt, answer = row.get(mapping["input"]), row.get(mapping["expected_output"])
        identifier = row.get(mapping["id"]) if mapping.get("id") else None
        if (isinstance(identifier, float) and math.isnan(identifier)) or str(identifier).strip().lower() in ("nan", "none", "null"): identifier = None  # id giữ chỗ, ví dụ chuỗi "NaN" ở OpenR1-Math
        metadata = {"hf_split": spec.get("split", "train"), "hf_row": index, **({"language": spec["language"]} if spec.get("language") else {})}
        record = {"id": f"{slug}-{identifier if identifier not in (None, '') else index}", "domain": spec.get("domain", "reasoning"), "task": spec.get("task", "sft"), "input": prompt, "expected_output": answer, "source": source, "license": dict(spec["license"]), "dataset_version": version, "metadata": metadata}
        if messages is not None: record["messages"] = messages
        for field in ("context", "reasoning"):
            if mapping.get(field) and row.get(mapping[field]) not in (None, ""): record[field] = row[mapping[field]]
        records.append(record)
    return records


def prepare_hf_sft(config: dict[str, Any], output_dir: str | Path | None = None, loader: RowLoader | None = None) -> dict[str, Any]:
    """Tải -> chuyển đổi -> kiểm tra/loại trùng/xuất qua build_dataset. Ghi ra raw.jsonl, train.jsonl, sft.jsonl, rejected.jsonl, manifest.json."""
    spec = config["hf_dataset"]; validate_spec(spec)
    version = config.get("dataset_version", "hf-v1"); output = Path(output_dir or config.get("output_dir", DEFAULT_SFT_DIR))
    raw = output / "raw.jsonl"; write_jsonl(raw, map_rows(load_hf_rows(spec, loader), spec, version))
    build_config = {"seed": config.get("seed", 0), "formats": config.get("formats", ["sft"]), "hf_dataset": {key: value for key, value in spec.items() if key != "token"}}
    return build_dataset([raw], output, version, build_config, config.get("eval_sources", []))


def load_preset(name: str, directory: str | Path = PRESET_DIR) -> dict[str, Any]:
    path = Path(directory) / f"{name}.json"
    if not path.is_file():
        available = ", ".join(sorted(item.stem for item in Path(directory).glob("*.json"))) or "(chưa có)"
        raise LookupError(f"Không có preset '{name}' trong {directory}; các preset hiện có: {available}")
    return json.loads(path.read_text(encoding="utf-8"))


def list_presets(directory: str | Path = PRESET_DIR) -> list[dict[str, Any]]:
    """Thông tin ngắn của mọi preset: tên, mô tả, dataset, revision, domain, ngôn ngữ, giấy phép và số dòng tối đa."""
    presets = []
    for path in sorted(Path(directory).glob("*.json")):
        config = json.loads(path.read_text(encoding="utf-8")); spec = config["hf_dataset"]; license_info = spec.get("license", {})
        presets.append({"name": path.stem, "description": config.get("description", ""), "dataset": spec["name"], "revision": spec.get("revision", "main"), "domain": spec.get("domain"), "language": spec.get("language"), "license": license_info.get("name"), "license_status": license_info.get("status"), "limit": spec.get("limit")})
    return presets


def parse_mix(values: list[str]) -> dict[str, float]:
    """Đọc danh sách `tên[:tỉ lệ]` (ví dụ code:0.5) thành tỉ lệ đã chuẩn hóa về tổng 1; bỏ trống tỉ lệ thì tính là 1."""
    weights: dict[str, float] = {}
    for value in values:
        name, _, weight = value.partition(":")
        if not name or name in weights: raise ValueError(f"Preset bị thiếu tên hoặc bị lặp: '{value}'")
        try: weights[name] = float(weight) if weight else 1.0
        except ValueError as error: raise ValueError(f"Tỉ lệ của preset '{name}' phải là số, không phải '{weight}'") from error
        if weights[name] <= 0: raise ValueError(f"Tỉ lệ của preset '{name}' phải lớn hơn 0")
    if not weights: raise ValueError("Hãy chọn ít nhất một preset")
    total = sum(weights.values())
    return {name: weight / total for name, weight in weights.items()}


def mix_counts(weights: dict[str, float], limits: dict[str, int], total: int | None = None) -> dict[str, int]:
    """Số dòng đọc từ mỗi preset theo tỉ lệ. Không ghi total thì lấy tổng lớn nhất mà không preset nào vượt limit của nó."""
    if total is None: total = int(min(limits[name] / weight for name, weight in weights.items()))
    counts = {name: max(1, round(total * weight)) for name, weight in weights.items()}
    over = [f"{name} (cần {count}, tối đa {limits[name]})" for name, count in counts.items() if count > limits[name]]
    if over: raise ValueError(f"Vượt số dòng tối đa của preset: {', '.join(over)}; hãy giảm --total hoặc tăng limit trong preset")
    return counts


def prepare_hf_mix(weights: dict[str, float], output_dir: str | Path, loader: RowLoader | None = None, total: int | None = None, limit: int | None = None,
                   version: str = "hf-mix-v1", seed: int = 17, directory: str | Path = PRESET_DIR) -> dict[str, Any]:
    """Trộn nhiều preset theo tỉ lệ thành một sft.jsonl. `limit` (nếu có) ghi đè số dòng tối đa của mọi preset."""
    presets = {name: load_preset(name, directory) for name in weights}
    for config in presets.values(): validate_spec(config["hf_dataset"])
    limits = {name: limit or config["hf_dataset"].get("limit") or 1000 for name, config in presets.items()}
    counts = mix_counts(weights, limits, total); records = []
    for name, config in presets.items():
        spec = {**config["hf_dataset"], "limit": counts[name]}
        records += map_rows(load_hf_rows(spec, loader), spec, version)
    output = Path(output_dir); raw = output / "raw.jsonl"; write_jsonl(raw, records)
    eval_sources = sorted({source for config in presets.values() for source in config.get("eval_sources", [])})
    mix = {name: {"weight": round(weights[name], 4), "rows": counts[name], "dataset": presets[name]["hf_dataset"]["name"], "revision": presets[name]["hf_dataset"].get("revision", "main")} for name in presets}
    return build_dataset([raw], output, version, {"seed": seed, "formats": ["sft"], "mix": mix}, eval_sources)
