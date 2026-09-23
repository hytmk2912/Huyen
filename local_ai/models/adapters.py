from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from local_ai.contracts import Message


class ModelCapability(StrEnum):
    CHAT = "chat"; REASONING = "reasoning"; CODING = "coding"; TOOL_CALLING = "tool_calling"; LONG_CONTEXT = "long_context"; VISION = "vision"; EMBEDDING = "embedding"; EVALUATION = "evaluation"


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
    quantization: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelConfig":
        return cls(**{**value, "capabilities": frozenset(ModelCapability(capability) for capability in value.get("capabilities", []))})

    @property
    def configuration_hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), default=lambda value: sorted(value), sort_keys=True).encode()).hexdigest()


class HuggingFaceModelAdapter:
    """Optional local Hugging Face runtime adapter; dependencies load only on use."""
    def __init__(self, config: ModelConfig): self.config, self.model, self.tokenizer = config, None, None
    @property
    def name(self) -> str: return self.config.name
    @property
    def capabilities(self) -> set[str]: return {capability.value for capability in self.config.capabilities}
    def load(self) -> None:
        if self.config.dtype == "fp8": raise RuntimeError("FP8 is not implemented for this adapter; select a tested runtime first")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error: raise RuntimeError("Install torch and transformers to load a Hugging Face model") from error
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(self.config.dtype)
        if dtype is None: raise ValueError(f"Unsupported dtype: {self.config.dtype}")
        kwargs = {"revision": self.config.revision, "local_files_only": self.config.offline}
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_source or self.config.source, revision=self.config.tokenizer_revision or self.config.revision, local_files_only=self.config.offline)
        self.model = AutoModelForCausalLM.from_pretrained(self.config.source, torch_dtype=dtype, device_map=self.config.device_map, **kwargs)
    def tokenizer_metadata(self) -> dict[str, str]:
        if self.tokenizer is None: raise RuntimeError("Load the model before requesting tokenizer metadata")
        source = self.config.tokenizer_source or self.config.source; revision = self.config.tokenizer_revision or self.config.revision
        payload = {"model_id": self.config.source, "model_revision": self.config.revision, "tokenizer_id": source, "tokenizer_revision": revision, "tokenizer_class": self.tokenizer.__class__.__name__}
        return {**payload, "configuration_hash": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()}
    def generate(self, messages: list[Message]) -> str:
        if self.model is None or self.tokenizer is None: self.load()
        prompt = self.tokenizer.apply_chat_template([{"role": message.role, "content": message.content} for message in messages], tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=512)
        return self.tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
