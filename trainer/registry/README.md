# Registry

This folder contains the registry system used by the framework to discover components by string key. For project-wide context, see [docs/README.md](../../docs/README.md).

## Purpose

- Provide a generic, typed `Registry[T]` with optional base-class enforcement
- Define the four global registries that the training loop uses to discover components
- Load user-defined plugin modules at startup to populate registries

## Layout

| File | Description |
|------|-------------|
| [`core.py`](core.py) | `Registry[T]` — generic registry with `register(key)` decorator, `get(key)`, `has(key)`, `keys()`, and optional base-class enforcement via `expected_base` |
| [`__init__.py`](__init__.py) | Instantiates the four global registries with their expected-base protocols |
| [`plugins.py`](plugins.py) | `load_plugins(modules)` — imports each module path once via `importlib.import_module()`. Tracks already-loaded modules to prevent double-registration |

## Contracts

The four global registries defined in [`__init__.py`](__init__.py):

| Registry | Expected base | Used for | Looked up via |
|----------|---------------|----------|---------------|
| `TRAINER_REGISTRY` | `Trainer` protocol | Training loop implementations | `train.trainer.type` |
| `DATASET_BUILDER_REGISTRY` | `DatasetFactory` protocol | Dataset construction factories | `data.datamodule.type` |
| `OPTIMIZER_BUILDER_REGISTRY` | `OptimizerFactory` protocol | Optimizer construction factories | Each entry's `type` in `model.component_optims` |
| `LOSS_BUILDER_REGISTRY` | `LossFactory` protocol | Loss function construction factories | `train.loss.type` |

Registration: `@<REGISTRY>.register("key")` on a class. Lookup: `<REGISTRY>.get("key")` returns the class (raises `KeyError` if missing).

Plugin loading is one-shot per process. Re-registration of the same key raises `KeyError`; this is intentional so that two plugins registering the same key produces a loud error rather than silent overwrite.

## How to extend

- Adding a new component type? Implement the appropriate protocol from [`../templates/`](../templates/) and register it. See [docs/07_extending.md](../../docs/07_extending.md).
- Adding a new *category* of registry (e.g. for metrics, callbacks)? Add a new global `Registry[T]` instantiation in [`__init__.py`](__init__.py) with the appropriate `expected_base`, then have a consumer call `<REGISTRY>.get(...)` somewhere in the pipeline.

## Cross-links

- Concepts: [docs/04_concepts.md § Registries](../../docs/04_concepts.md#registries) and [§ Plugins](../../docs/04_concepts.md#plugins)
- Extending recipes: [docs/07_extending.md](../../docs/07_extending.md)
- Hub: [docs/README.md](../../docs/README.md)

## Gotchas / invariants

- Registering the same key twice raises `KeyError`. See [`core.py`](core.py).
- Base-class enforcement uses `issubclass(cls, expected_base)`. For `@runtime_checkable` protocols with method-only members (as the four template protocols are), this is a structural check that passes when the class declares matching method names.
- Plugin modules are imported exactly once per process. The `_LOADED_MODULES` set in [`plugins.py`](plugins.py) prevents re-imports, which would otherwise re-run `@register` decorators and produce duplicate-key errors.
