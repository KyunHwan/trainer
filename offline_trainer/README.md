# offline_trainer/

## What this folder is

`offline_trainer/` is the main Python package for this repository. It wires together:

- YAML config resolution (via the vendored `policy_constructor/model_constructor`)
- component instantiation from a registry (trainer/data/IO/loss/optim/scheduler/etc.)
- a minimal training loop (single-device)

## What a beginner should do here

- Start with the repo root `README.md` for the end-to-end workflow.
- If you want to add custom components, you usually **do not** edit this package; instead add a plugin under `extensions/` and reference it from YAML `imports:`.

## Key files

- `offline_trainer/experiment.py`: main entrypoint `run_experiment(config_path)`.
- `offline_trainer/run.py`: thin wrapper around `run_experiment()`.
- `offline_trainer/registry/`: builds the registry and registers built-in keys.
- `offline_trainer/components/`: reference implementations of pluggable components.
- `offline_trainer/engine/`: training loop + checkpoint/accelerator utilities.
- `offline_trainer/config/`: config validation + error formatting.
- `offline_trainer/deps/`: dependency shim for the model-constructor library.

## How it connects to the pipeline

`run_experiment()` (in `offline_trainer/experiment.py`) follows this high-level flow:

1. `resolve_config()` → merged YAML mapping + source locations
2. `validate_preflight()` → checks config structure and spec prefixes
3. `build_registry()` → registers model blocks + training components
4. `build_model()` → constructs `torch.nn.Module` and executes YAML `imports:`
5. `instantiate_value()` → builds trainer/data/IO/loss/optim/etc. from `_type_` specs
6. `trainer.fit(...)` → runs the loop

## Common edits / extension points

- Add a new Trainer/DataModule/Loss/etc.: implement it anywhere importable (commonly `extensions/`) and register it via a `register(registry)` hook.
- Add new built-ins: edit `offline_trainer/registry/builtins.py` (less common; this is “core”).

## Links

- [Root quickstart](../README.md#quickstart)
- [Architecture tour](../README.md#architecture-tour)
- [Config overview](../README.md#config-system-what-yaml-keys-mean)
- [Plugins](../README.md#registry--plugins-how-yaml-selects-components)
