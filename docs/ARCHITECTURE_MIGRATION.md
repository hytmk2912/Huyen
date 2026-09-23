# Architecture Migration

## Old architecture
The repository root contained independent nanoGPT-style 3B/7B PyTorch and DeepSpeed training scripts, RoPE model definitions, shell orchestration, GPT-2/tiktoken data preparation, and Vietnamese operational documents. They target randomly initialized custom models and have separate, incompatible configuration and tokenizer assumptions.

## New architecture
`local_ai` is the canonical application package. `configs/models`, `configs/datasets`, `configs/training`, and `configs/experiments` are its canonical configuration roots. The primary configurable model is the verified Hugging Face identifier `huihui-ai/Huihui-Qwen3.8-27B-abliterated`; model selection is capability-based. The corpus system uses the selected model tokenizer, persistent state, and verified remote-upload accounting.

## Files to keep
Keep `local_ai/`, `configs/`, `data/`, `tests/`, and the new README. Keep legacy scripts only under `archive/legacy-nanogpt/` as non-production research history; no checkpoint or dataset is deleted.

## Files to migrate
`train_*.py`, model definitions, DeepSpeed JSON, orchestration shell scripts, and old operational documents move to `archive/legacy-nanogpt/`. Their useful historical settings may be consulted, but are not supported by the canonical runtime.

## Files to delete
No data, checkpoints, or experiment records are deleted. Root-level legacy entry points are removed from the production surface by archival because they conflict with the selected Hugging Face model and tokenizer.

## Risks
The primary model requires optional `transformers`/`torch` dependencies and suitable hardware. Its exact revision must be pinned before a reproducible training run. Existing byte and cl100k smoke artifacts are non-production and cannot contribute to official corpus totals.

## Test plan
Run unit tests and compilation. Test model configuration/router with scripted adapters in CI. Model loading and authenticated Hugging Face upload tests skip unless dependencies, hardware, and credentials are provided. Before acquisition, run corpus tokenizer-info against the configured primary model and one authenticated shard upload.
