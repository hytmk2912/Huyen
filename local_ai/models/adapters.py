"""Cấu hình model và adapter chạy model Hugging Face cục bộ."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from local_ai.contracts import Message

QUANTIZATIONS = ("nf4",)


class ModelCapability(StrEnum):
    CHAT = "chat"
    REASONING = "reasoning"
    CODING = "coding"
    TOOL_CALLING = "tool_calling"
    LONG_CONTEXT = "long_context"
    VISION = "vision"
    EMBEDDING = "embedding"
    EVALUATION = "evaluation"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    source: str
    revision: str = "main"
    tokenizer_source: str | None = None
    tokenizer_revision: str | None = None
    dtype: str = "bfloat16"
    device_map: str | dict[str, Any] = "auto"
    capabilities: frozenset[ModelCapability] = field(default_factory=frozenset)
    offline: bool = False
    # "nf4": nạp trọng số 4-bit (bitsandbytes) khi chạy suy luận; huấn luyện 4-bit chọn bằng method "qlora".
    quantization: str | None = None
    # Số tham số (tỷ), chỉ để ước lượng bộ nhớ GPU cần cho trọng số.
    params_billion: float | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelConfig":
        capabilities = frozenset(ModelCapability(capability) for capability in value.get("capabilities", []))
        config = cls(**{**value, "capabilities": capabilities})
        if config.quantization not in (None, *QUANTIZATIONS):
            raise ValueError(
                f"Model '{config.name}': không hỗ trợ quantization {config.quantization}; dùng nf4 hoặc bỏ trống"
            )
        return config

    @property
    def configuration_hash(self) -> str:
        payload = json.dumps(asdict(self), default=lambda value: sorted(value), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()


def quantization_settings(kind: str) -> dict[str, Any]:
    """Tham số BitsAndBytesConfig cho 4-bit NF4 (dùng chung cho QLoRA và suy luận 4-bit).
    bnb_4bit_compute_dtype được gán torch.bfloat16 lúc nạp model."""
    if kind != "nf4":
        raise ValueError(f"Không hỗ trợ quantization: {kind}")
    return {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "bfloat16",
    }


class HuggingFaceModelAdapter:
    """Adapter tùy chọn để chạy model Hugging Face cục bộ; thư viện chỉ được nạp khi dùng tới."""

    def __init__(self, config: ModelConfig):
        self.config, self.model, self.tokenizer = config, None, None

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def capabilities(self) -> set[str]:
        return {capability.value for capability in self.config.capabilities}

    def load(self) -> None:
        if self.config.dtype == "fp8":
            raise RuntimeError("Adapter này chưa hỗ trợ FP8; hãy chọn một runtime đã được thử nghiệm trước")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("Hãy cài torch và transformers để nạp model Hugging Face") from error
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(self.config.dtype)
        if dtype is None:
            raise ValueError(f"Không hỗ trợ dtype: {self.config.dtype}")
        kwargs: dict[str, Any] = {
            "revision": self.config.revision,
            "local_files_only": self.config.offline,
            "torch_dtype": dtype,
        }
        if self.config.quantization:
            from transformers import BitsAndBytesConfig

            settings = quantization_settings(self.config.quantization)
            settings["bnb_4bit_compute_dtype"] = torch.bfloat16
            kwargs["quantization_config"] = BitsAndBytesConfig(**settings)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_source or self.config.source,
            revision=self.config.tokenizer_revision or self.config.revision,
            local_files_only=self.config.offline,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.source, device_map=self.config.device_map, **kwargs
        )

    def tokenizer_metadata(self) -> dict[str, str]:
        if self.tokenizer is None:
            raise RuntimeError("Hãy nạp model trước khi lấy thông tin tokenizer")
        source = self.config.tokenizer_source or self.config.source
        revision = self.config.tokenizer_revision or self.config.revision
        payload = {
            "model_id": self.config.source,
            "model_revision": self.config.revision,
            "tokenizer_id": source,
            "tokenizer_revision": revision,
            "tokenizer_class": self.tokenizer.__class__.__name__,
        }
        return {
            **payload,
            "configuration_hash": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
        }

    def generate(self, messages: list[Message]) -> str:
        if self.model is None or self.tokenizer is None:
            self.load()
        prompt = self.tokenizer.apply_chat_template(
            [{"role": message.role, "content": message.content} for message in messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=512)
        return self.tokenizer.decode(output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
