# Local Autonomous AI Architecture

## Scope
This repository provides orchestration infrastructure, not a trained model or live market/web integration. Model weights, retrieval indexes, datasets, and credentials remain external and are selected through configuration.

## Layers

1. **Configuration and reproducibility**: JSON configurations are loaded into immutable run settings. `seed_everything` records and seeds Python randomness; `RunTracker` writes config, metrics, events, and checkpoint references to a run directory.
2. **Model access**: `ModelAdapter` is the backend-neutral protocol. `ScriptedModelAdapter` is a deterministic local development adapter. Add adapters for local inference engines without changing the agent.
3. **Routing**: `ModelRouter` chooses a configured model by capability, with an optional default fallback.
4. **Agent loop**: `AutonomousAgent` runs request → plan → tool selection → execution → observation → evaluation → correction/retry → final response. Planning and evaluation are intentionally structured interfaces so a model or deterministic policy can be substituted.
5. **Tools and sandboxing**: `ToolRegistry` exposes declared tools. `PythonSandbox` runs supplied code in a temporary working directory, with timeout and output capture. It is an isolation boundary interface—not a security guarantee against hostile code; production deployments must use OS/container isolation.
6. **Context and knowledge**: `MemoryStore` keeps bounded conversation records. `Retriever` is a protocol for external/RAG knowledge, keeping frequently changing facts out of weights.
7. **Data, training, and evaluation**: Dataset manifests capture version/source/split metadata. Training plans support future SFT, preference optimization, and RFT runs. `local_ai.data.hub` maps Hugging Face dataset rows onto the dataset schema before the standard build; `local_ai.training.finetune` is an optional SFT harness (full or LoRA) that skips when GPU or libraries are absent. Benchmarks use a common evaluator and distinct capability suites.

## Extension points
- Implement `ModelAdapter.generate` for a local backend (for example a local server or in-process runtime).
- Register capability-scoped models in `models` configuration.
- Register audited tools via `ToolRegistry`; future interfaces include coding, reasoning, research, trading, retrieval, terminal execution, and file operations.
- Implement `Retriever.search` for a vector index or live data provider.
- `local_ai.training.finetune` is the first concrete trainer behind `TrainingPlan`; preference optimization and RFT remain future work.

## Safety and limits
Tool access is explicit, bounded by iteration and timeout limits, and all tool results are observed before the next decision. The included demo uses only a deterministic model and calculator tool. Trading adapters are analysis interfaces only and do not place orders.
