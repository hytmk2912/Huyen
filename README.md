# Local Autonomous AI Platform

A configuration-driven local AI platform. The verified primary model identifier is `huihui-ai/Huihui-Qwen3.8-27B-abliterated`; additional local models are listed in `configs/models/platform.json` and selected by capability through the model router. This repository does not train or download the 27B model by default.

## Architecture
- `local_ai.models`: model configuration, adapters, tokenizer access, and capability routing.
- `local_ai.data`: one canonical corpus/data system with source provenance, filtering, tokenization, shards, registry, and optional Hugging Face storage.
- `local_ai.agents`: bounded plan/tool/observe/evaluate/retry agent loop.
- `local_ai.tools`: explicit tool registry and development sandbox interface.
- `local_ai.training`: training plans and the optional SFT harness (`finetune.py`, full or LoRA).
- `local_ai.evaluation`, `local_ai.experiments`: configuration-driven future training, evaluation, and reproducibility primitives.

## Corpus
The target is **10,000,000,000,000 real tokens**. It is only a target: local or configured tokens never count as uploaded corpus progress. Only remotely verified production shards count. Source metadata and licenses are required. No large acquisition starts automatically.

The production tokenizer must be loaded from the configured primary model. The old byte and `cl100k_base` smoke tokenizers are test-only and cannot count toward the production corpus.

## Models
`configs/models/platform.json` holds a list of models (`name`, `source` Hugging Face id, `dtype`, `capabilities`, optional revision/tokenizer/device settings). The first entry is the primary model. Add a model by appending an entry with a unique `name`; the router and the training harness reference models by name.

## Training (SFT)
Training is an optional scaffold: nothing trains by default and no model or dataset is downloaded by tests.

1. Prepare data from a Hugging Face dataset. Edit `configs/datasets/hf_sft.json`: set `hf_dataset.name`, the license from the dataset card, and either `messages_field` (chat column) or `mapping.input` / `mapping.expected_output`. Then:

   ```bash
   python -m pip install datasets
   python -m local_ai.data hf-sft --config configs/datasets/hf_sft.json [--dataset org/name] [--limit 1000]
   ```

   Rows are mapped onto the dataset schema and go through the existing validation, deduplication and eval-overlap checks. Output: `raw.jsonl`, `train.jsonl`, `sft.jsonl`, `rejected.jsonl`, `manifest.json` under `output_dir`. `HF_TOKEN` is read from the environment only (needed for gated/private datasets).

2. Fine-tune. `configs/training/sft.json` names the base model (from the model list), the `sft.jsonl` path, the output directory and hyperparameters. `method` is `full` or `lora`; gradient checkpointing is on by default.

   ```bash
   python -m pip install torch transformers trl peft datasets
   python -m local_ai.training.finetune --config configs/training/sft.json --dry-run
   python -m local_ai.training.finetune --config configs/training/sft.json --method lora
   ```

   Checkpoints are written as `checkpoint-N` in the output directory; a rerun resumes from the latest one (`--no-resume` to start over). The final LoRA adapter is saved to `adapter/`, a full fine-tune to `final/`. When a GPU or an optional library is missing the run reports `"status": "skipped"` instead of failing.

## Setup and checks
Install optional local-model dependencies when model loading is needed:

```bash
python -m pip install transformers torch huggingface_hub
# training extras (optional): trl peft datasets
python -m unittest discover -s tests -v
python -m compileall -q local_ai
```

Use `HF_TOKEN` only in the environment (for upload or gated dataset download); set `HF_DATASET_REPO` there or in non-secret configuration. Never commit credentials. Run `python -m local_ai.data secret-scan` before acquisition.

## Roadmap
1. Pin and smoke-test the primary model/tokenizer on target hardware.
2. Complete authenticated one-shard remote verification.
3. Add reviewed, licensed corpus source adapters incrementally.
4. Implement distributed BF16 training; FP8 remains unsupported until a selected runtime and hardware are tested.
5. Expand agent tools, isolation, evaluation suites, and recovery testing.
