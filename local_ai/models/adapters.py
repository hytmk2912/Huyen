from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
import urllib.parse
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from local_ai.contracts import Message

MODEL_KINDS = ("text", "multimodal")
BACKENDS = ("transformers", "openai_compatible")
QUANTIZATIONS = ("4bit", "8bit")
TORCH_DTYPES = ("bfloat16", "float16", "float32")


COMMIT = re.compile(r"^[0-9a-f]{40}$")  # revision ghim theo mã commit đầy đủ của repo trên Hugging Face
HF_REPO_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")  # tên repo trên Hugging Face (org/name); tên khác (ví dụ "scripted") không tải từ Hub
ALLOW_PICKLE_FLAG = "--allow-pickle-weights"
_unsafe = {"allow_pickle_weights": False}  # chỉ đổi qua cờ dòng lệnh --allow-pickle-weights (set_allow_pickle_weights)


class UnsafeModelError(RuntimeError):
    """Từ chối nạp model/adapter vì không an toàn: revision chưa ghim, chỉ có trọng số dạng pickle (.bin/.pkl) mà chưa xác nhận."""


def set_allow_pickle_weights(allowed: bool) -> None:
    """Gọi từ lệnh có cờ --allow-pickle-weights: cho phép nạp trọng số .bin/.pkl (pickle có thể chạy mã độc khi nạp)."""
    _unsafe["allow_pickle_weights"] = bool(allowed)
    if allowed: print("CẢNH BÁO: đã bật --allow-pickle-weights. Trọng số dạng pickle (.bin/.pkl) có thể chứa mã độc chạy ngay khi nạp; "
                      "chỉ dùng với repo bạn tin tưởng.", file=sys.stderr, flush=True)


def is_local_source(source: str) -> bool:
    return Path(source).exists()


def unpinned_message(source: str, revision: str | None, what: str = "model") -> str | None:
    """Lời báo nếu model/tokenizer tải từ Hugging Face chưa ghim revision bằng mã commit 40 ký tự; thư mục trên máy thì không cần ghim."""
    if is_local_source(source) or not HF_REPO_ID.match(source) or COMMIT.match(revision or ""): return None
    return (f"{what} '{source}' chưa ghim revision (đang là '{revision}'). Hãy ghi mã commit 40 ký tự của repo trên Hugging Face "
            "vào trường revision trong configs/models/platform.json, để model không bị thay âm thầm.")


def check_pinned(source: str, revision: str | None, what: str = "model") -> None:
    """Lúc nạp: chưa ghim thì cảnh báo. Bắt buộc ghim được kiểm tra khi đọc danh sách model (local_ai.config.settings.load_model_configs)."""
    message = unpinned_message(source, revision, what)
    if message: print("Cảnh báo: " + message, file=sys.stderr, flush=True)


def safe_model_kwargs(source: str, revision: str | None) -> dict[str, Any]:
    """Tham số an toàn cho from_pretrained của model: không chạy code từ repo (trust_remote_code=False), chỉ nạp safetensors
    (trừ khi đã bật --allow-pickle-weights)."""
    check_pinned(source, revision)
    return {"trust_remote_code": False, "use_safetensors": not _unsafe["allow_pickle_weights"]}


def safe_tokenizer_kwargs(source: str, revision: str | None) -> dict[str, Any]:
    """Tham số an toàn cho from_pretrained của tokenizer/processor: không chạy code từ repo."""
    check_pinned(source, revision, "tokenizer")
    return {"trust_remote_code": False}


def from_pretrained_safely(loader: Any, source: str, **kwargs: Any) -> Any:
    """Gọi loader.from_pretrained với tham số an toàn. Repo không có safetensors thì báo lỗi tiếng Việt, chỉ cách xác nhận bằng cờ."""
    kwargs = {**kwargs, **safe_model_kwargs(source, kwargs.get("revision"))}
    try:
        return loader.from_pretrained(source, **kwargs)
    except (OSError, EnvironmentError) as error:
        if kwargs["use_safetensors"] and "safetensors" in str(error).lower():
            raise UnsafeModelError(f"Model '{source}' không có file .safetensors (có thể chỉ có .bin/.pkl dạng pickle, có thể chứa mã độc). "
                                   f"Chỉ khi tin tưởng repo này mới chạy lại lệnh với cờ {ALLOW_PICKLE_FLAG}.") from error
        raise


def check_adapter_files(adapter_dir: str | Path) -> None:
    """Adapter LoRA phải có adapter_model.safetensors; chỉ có adapter_model.bin (pickle) thì cần cờ --allow-pickle-weights."""
    folder = Path(adapter_dir)
    if (folder / "adapter_model.safetensors").is_file(): return
    if (folder / "adapter_model.bin").is_file() and not _unsafe["allow_pickle_weights"]:
        raise UnsafeModelError(f"Adapter trong {folder} chỉ có adapter_model.bin (pickle, có thể chứa mã độc). "
                               f"Chỉ khi tin tưởng nguồn của adapter mới chạy lại lệnh với cờ {ALLOW_PICKLE_FLAG}.")


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
    device_map: str | dict[str, Any] | None = "auto"  # null: không truyền device_map (chạy trên CPU không cần accelerate)
    capabilities: frozenset[ModelCapability] = field(default_factory=frozenset)
    offline: bool = False
    quantization: str | None = None
    kind: str = "text"
    params_b: float | None = None
    backend: str = "transformers"  # "openai_compatible": gọi server local (Ollama, llama.cpp, vLLM); source là tên model trên server
    base_url: str | None = None  # ví dụ http://localhost:11434/v1
    api_key_env: str | None = None  # tên biến môi trường chứa khóa của server (nếu server cần); không ghi khóa vào cấu hình
    timeout_s: float = 120.0
    adapter_path: str | None = None  # thư mục LoRA adapter đã train (ví dụ .runs/sft/adapter), nạp chồng lên model gốc
    max_new_tokens: int = 512
    temperature: float = 0.0  # 0: greedy (do_sample=False), chạy lại ra đúng câu trả lời cũ; lớn hơn 0 thì lấy mẫu ngẫu nhiên với seed cố định
    seed: int = 0  # seed khi lấy mẫu (temperature > 0); model qua server nhận seed này trong yêu cầu
    enable_thinking: bool | None = None  # False: tắt chế độ suy nghĩ <think> của Qwen3 (truyền vào chat template); None: để mặc định của template

    def __post_init__(self) -> None:
        if self.kind not in MODEL_KINDS: raise ValueError(f"Model '{self.name}': kind phải là {' hoặc '.join(MODEL_KINDS)}, không phải '{self.kind}'")
        if self.quantization is not None and self.quantization not in QUANTIZATIONS: raise ValueError(f"Model '{self.name}': quantization phải là {' hoặc '.join(QUANTIZATIONS)} (hoặc bỏ trống), không phải '{self.quantization}'")
        if self.params_b is not None and (isinstance(self.params_b, bool) or not isinstance(self.params_b, (int, float)) or self.params_b <= 0): raise ValueError(f"Model '{self.name}': params_b phải là số tỷ tham số lớn hơn 0")
        if ModelCapability.VISION in self.capabilities and self.kind != "multimodal": raise ValueError(f"Model '{self.name}' có khả năng vision thì kind phải là 'multimodal'")
        if self.backend not in BACKENDS: raise ValueError(f"Model '{self.name}': backend phải là {' hoặc '.join(BACKENDS)}, không phải '{self.backend}'")
        if self.backend == "openai_compatible":
            url = urllib.parse.urlsplit(self.base_url or "")
            if url.scheme not in ("http", "https") or not url.hostname: raise ValueError(f"Model '{self.name}': backend openai_compatible cần base_url dạng http://máy:cổng/v1, không phải '{self.base_url}'")
        if isinstance(self.timeout_s, bool) or not isinstance(self.timeout_s, (int, float)) or self.timeout_s <= 0: raise ValueError(f"Model '{self.name}': timeout_s phải là số giây lớn hơn 0")
        if isinstance(self.max_new_tokens, bool) or not isinstance(self.max_new_tokens, int) or self.max_new_tokens <= 0: raise ValueError(f"Model '{self.name}': max_new_tokens phải là số nguyên lớn hơn 0")
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)) or self.temperature < 0: raise ValueError(f"Model '{self.name}': temperature phải là số lớn hơn hoặc bằng 0 (0 là greedy)")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0: raise ValueError(f"Model '{self.name}': seed phải là số nguyên không âm")
        if self.enable_thinking is not None and not isinstance(self.enable_thinking, bool): raise ValueError(f"Model '{self.name}': enable_thinking phải là true, false hoặc bỏ trống")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelConfig":
        return cls(**{**value, "capabilities": frozenset(ModelCapability(capability) for capability in value.get("capabilities", []))})

    @property
    def generation_settings(self) -> dict[str, Any]:
        """Cách sinh chữ (ghi vào cài đặt và báo cáo chấm): greedy hay lấy mẫu, temperature, seed, có tắt chế độ suy nghĩ không."""
        return {"do_sample": self.temperature > 0, "temperature": self.temperature, "seed": self.seed, "enable_thinking": self.enable_thinking}

    @property
    def configuration_hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), default=lambda value: sorted(value), sort_keys=True).encode()).hexdigest()


def resolve_torch_dtype(torch: Any, name: str) -> Any:
    """Đổi tên dtype sang kiểu của torch; GPU không có bf16 (ví dụ T4) thì tự chuyển sang fp16 và in cảnh báo."""
    if name == "fp8": raise RuntimeError("Adapter này chưa hỗ trợ FP8; hãy chọn một runtime đã được thử nghiệm trước")
    if name not in TORCH_DTYPES: raise ValueError(f"Không hỗ trợ dtype: {name}")
    if name == "bfloat16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        print("Cảnh báo: GPU này không hỗ trợ bfloat16 (ví dụ T4), tự chuyển sang float16.", file=sys.stderr)
        return torch.float16
    return getattr(torch, name)


def quantization_config(transformers: Any, quantization: str | None, compute_dtype: Any) -> Any:
    """Tạo BitsAndBytesConfig cho '4bit' (NF4, kiểu QLoRA) hoặc '8bit'; None nếu không nén."""
    if quantization is None: return None
    if quantization not in QUANTIZATIONS: raise ValueError(f"quantization phải là {' hoặc '.join(QUANTIZATIONS)}, không phải '{quantization}'")
    if importlib.util.find_spec("bitsandbytes") is None: raise RuntimeError(f"Nén {quantization} cần thư viện bitsandbytes; hãy chạy `python -m pip install bitsandbytes`")
    if quantization == "8bit": return transformers.BitsAndBytesConfig(load_in_8bit=True)
    return transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype)


def model_loader_class(transformers: Any, kind: str) -> Any:
    """Lớp dùng để nạp model: AutoModelForMultimodalLM (bản cũ: AutoModelForImageTextToText) cho model multimodal, AutoModelForCausalLM cho model text."""
    if kind == "text": return transformers.AutoModelForCausalLM
    for name in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText"):
        loader = getattr(transformers, name, None)
        if loader is not None: return loader
    raise RuntimeError("Bản transformers đang cài không có AutoModelForMultimodalLM hay AutoModelForImageTextToText; hãy nâng cấp: `python -m pip install -U transformers`")


def load_pretrained(config: ModelConfig) -> tuple[Any, Any]:
    """Nạp (model, processor hoặc tokenizer) theo loại model và kiểu nén đã khai báo; chỉ import thư viện khi gọi."""
    try:
        import torch
        import transformers
    except ImportError as error: raise RuntimeError("Hãy cài torch và transformers để nạp model Hugging Face") from error
    dtype = resolve_torch_dtype(torch, config.dtype)
    processor_class = transformers.AutoProcessor if config.kind == "multimodal" else transformers.AutoTokenizer
    tokenizer_source, tokenizer_revision = config.tokenizer_source or config.source, config.tokenizer_revision or config.revision
    processor = processor_class.from_pretrained(tokenizer_source, revision=tokenizer_revision, local_files_only=config.offline, **safe_tokenizer_kwargs(tokenizer_source, tokenizer_revision))
    kwargs = {"revision": config.revision, "local_files_only": config.offline, "dtype": dtype, "device_map": config.device_map}
    if config.quantization: kwargs["quantization_config"] = quantization_config(transformers, config.quantization, dtype)
    model = from_pretrained_safely(model_loader_class(transformers, config.kind), config.source, **kwargs)
    if config.adapter_path:
        try:
            from peft import PeftModel
        except ImportError as error: raise RuntimeError("Nạp LoRA adapter cần thư viện peft; hãy chạy `python -m pip install peft`") from error
        if not Path(config.adapter_path).is_dir(): raise FileNotFoundError(f"Không tìm thấy thư mục adapter: {config.adapter_path}")
        check_adapter_files(config.adapter_path)
        model = PeftModel.from_pretrained(model, config.adapter_path)
    return model, processor


def chat_messages(config: ModelConfig, messages: list[Message]) -> list[dict[str, Any]]:
    """Đổi Message sang dạng chat template; ảnh chỉ gửi được tới model multimodal."""
    if config.kind != "multimodal":
        if any(message.images for message in messages): raise ValueError(f"Model '{config.name}' chỉ nhận chữ (kind='text'), không gửi ảnh được; hãy dùng model multimodal")
        return [{"role": message.role, "content": message.content} for message in messages]
    result = []
    for message in messages:
        for image in message.images:
            if not Path(image).is_file(): raise FileNotFoundError(f"Không tìm thấy file ảnh: {image}")
        result.append({"role": message.role, "content": [{"type": "image", "path": str(image)} for image in message.images] + [{"type": "text", "text": message.content}]})
    return result


class HuggingFaceModelAdapter:
    """Adapter tùy chọn để chạy model Hugging Face cục bộ; thư viện chỉ được nạp khi dùng tới."""
    def __init__(self, config: ModelConfig): self.config, self.model, self.tokenizer = config, None, None
    @property
    def name(self) -> str: return self.config.name
    @property
    def capabilities(self) -> set[str]: return {capability.value for capability in self.config.capabilities}
    def load(self) -> None:
        # Với model multimodal, self.tokenizer là processor (gồm cả tokenizer và phần xử lý ảnh).
        self.model, self.tokenizer = load_pretrained(self.config)
    def tokenizer_metadata(self) -> dict[str, str]:
        if self.tokenizer is None: raise RuntimeError("Hãy nạp model trước khi lấy thông tin tokenizer")
        source = self.config.tokenizer_source or self.config.source; revision = self.config.tokenizer_revision or self.config.revision
        payload = {"model_id": self.config.source, "model_revision": self.config.revision, "tokenizer_id": source, "tokenizer_revision": revision, "tokenizer_class": self.tokenizer.__class__.__name__}
        return {**payload, "configuration_hash": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()}
    def template_kwargs(self) -> dict[str, Any]:
        """Tham số thêm cho chat template: enable_thinking khi cấu hình có đặt (template không dùng biến này thì bỏ qua)."""
        return {} if self.config.enable_thinking is None else {"enable_thinking": self.config.enable_thinking}
    def generation_kwargs(self) -> dict[str, Any]:
        """Tham số sinh chữ: temperature 0 thì greedy (ghi đè cấu hình lấy mẫu mặc định của model, ví dụ Qwen đặt do_sample=True);
        lớn hơn 0 thì lấy mẫu sau khi đặt seed, để chạy lại vẫn ra cùng kết quả."""
        if self.config.temperature == 0: return {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
        import torch
        torch.manual_seed(self.config.seed)
        return {"do_sample": True, "temperature": float(self.config.temperature)}
    def generate(self, messages: list[Message]) -> str:
        conversation = chat_messages(self.config, messages)
        if self.model is None or self.tokenizer is None: self.load()
        if self.config.kind == "multimodal":
            inputs = self.tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", **self.template_kwargs()).to(self.model.device)
        else:
            prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, **self.template_kwargs())
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=self.config.max_new_tokens, **self.generation_kwargs())
        return self.tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
