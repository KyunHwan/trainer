# Modeling

This folder contains the model-construction adapter that bridges the trainer framework with [policy_constructor](../../policy_constructor/). For project-wide context, see [docs/README.md](../../docs/README.md).

## Purpose

- Provide a `ModelFactory` protocol for building PyTorch modules from a config dict
- Implement `PolicyConstructorModelFactory` which calls `model_constructor.build_model()` for each component

## Layout

| File | Description |
|------|-------------|
| [`factories.py`](factories.py) | `ModelFactory` protocol and `PolicyConstructorModelFactory`. Supports three build modes: (1) single model via `config_path` key, (2) inline config via `config` key, (3) **multiple named components via individual keys** (the standard experiment pattern — input is `{name: yaml_path}`, output is `{name: nn.Module}`) |

## Contracts

`PolicyConstructorModelFactory.build(model_cfg)`:

- If `model_cfg` has a `"config_path"` key → return `build_model(model_cfg["config_path"])` — a single `nn.Module`.
- Else if `model_cfg` has a `"config"` key → return `build_model(model_cfg["config"])` — a single `nn.Module` from an inline dict.
- Else (the standard experiment case) → iterate the dict, call `build_model(v)` for each value, return `{k: nn.Module}` keyed by component name.

The training entrypoints always use case (3): they call `_build_models` which passes `config.model.component_config_paths.as_dict()` (after resolving relative paths against the project root) into `factory.build(...)`.

## How to extend

The factory is instantiated directly inside `_build_models` (no registry indirection); swap in a different factory by editing the entrypoint. If you want to support a different model-construction backend (e.g. an OmegaConf-driven one), implement the `ModelFactory` protocol and plug it in there.

## Cross-links

- Concepts: [docs/04_concepts.md § Factories](../../docs/04_concepts.md#factories) and [§ The nn.ModuleDict convention](../../docs/04_concepts.md#the-nnmoduledict-convention)
- Submodule schema: [`policy_constructor/README.md`](../../policy_constructor/README.md) and `policy_constructor/model_constructor/config/` (do not re-document its schema here)
- Hub: [docs/README.md](../../docs/README.md)

## Gotchas / invariants

- If `model_constructor` is not importable, `factories.py` falls back to adding `policy_constructor/` to `sys.path` at import time. So the submodule just needs to be present on disk for the factory to work.
- Relative config paths in `component_config_paths` are resolved against the project root (parent of `trainer/`) by `_build_models()`, not against the working directory. Absolute paths are also supported and pass through unchanged.
