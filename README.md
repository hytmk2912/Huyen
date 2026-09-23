# Local Autonomous AI Platform

A configuration-driven local AI platform. The verified primary model identifier is `huihui-ai/Huihui-Qwen3.8-27B-abliterated`; additional local models are selected by capability through the model router. This repository does not train or download the 27B model by default.

## Architecture
- `local_ai.models`: model configuration, adapters, tokenizer access, and capability routing.
- `local_ai.data`: one canonical corpus/data system with source provenance, filtering, tokenization, shards, registry, and optional Hugging Face storage.
- `local_ai.agents`: bounded plan/tool/observe/evaluate/retry agent loop.
- `local_ai.tools`: explicit tool registry and development sandbox interface.
- `local_ai.training`, `local_ai.evaluation`, `local_ai.experiments`: configuration-driven future training, evaluation, and reproducibility primitives.

## Corpus
The target is **10,000,000,000,000 real tokens**. It is only a target: local or configured tokens never count as uploaded corpus progress. Only remotely verified production shards count. Source metadata and licenses are required. No large acquisition starts automatically.

The production tokenizer must be loaded from the configured primary model. The old byte and `cl100k_base` smoke tokenizers are test-only and cannot count toward the production corpus.

## Setup and checks
Install optional local-model dependencies when model loading is needed:

```bash
python -m pip install transformers torch huggingface_hub
python -m unittest discover -s tests -v
python -m compileall -q local_ai
```

Use `HF_TOKEN` only in the environment when upload is enabled; set `HF_DATASET_REPO` there or in non-secret configuration. Never commit credentials. Run `python -m local_ai.data secret-scan` before acquisition.

## Roadmap
1. Pin and smoke-test the primary model/tokenizer on target hardware.
2. Complete authenticated one-shard remote verification.
3. Add reviewed, licensed corpus source adapters incrementally.
4. Implement distributed BF16 training; FP8 remains unsupported until a selected runtime and hardware are tested.
5. Expand agent tools, isolation, evaluation suites, and recovery testing.
