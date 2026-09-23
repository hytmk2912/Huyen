from __future__ import annotations

from local_ai.contracts import Message, ModelAdapter
from local_ai.models.adapters import ModelConfig


class ScriptedModelAdapter:
    def __init__(self, name: str, responses: list[str], capabilities: set[str] | None = None):
        self.name, self.responses = name, iter(responses)
        self.capabilities = capabilities or {"chat", "reasoning", "tool_calling"}
    def generate(self, messages: list[Message]) -> str: return next(self.responses)


class ModelRouter:
    def __init__(self, models: list[ModelAdapter], default: str | None = None):
        self._models = {model.name: model for model in models}; self._default = default
    def select(self, capability: str) -> ModelAdapter:
        for model in self._models.values():
            if capability in model.capabilities: return model
        if self._default and self._default in self._models: return self._models[self._default]
        raise LookupError(f"No model configured for capability: {capability}")
    @classmethod
    def from_configs(cls, configs: list[ModelConfig], factory): return cls([factory(config) for config in configs])
