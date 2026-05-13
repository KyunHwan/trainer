# Config

This folder contains the YAML-driven configuration system with composition support and Pydantic validation. For project-wide context, see [docs/README.md](../../docs/README.md).

## Purpose

- Load experiment configs from YAML with `defaults` composition and deep merge
- Validate configs against a typed Pydantic schema (`ExperimentConfig`)
- Produce structured, actionable error messages when validation fails

## Layout

| File | Description |
|------|-------------|
| [`loader.py`](loader.py) | `load_config(path)` — recursive YAML loader. Supports `defaults: [{key: relative_path}]` for config composition. Resolves relative paths against the config file's directory. Deep-merges defaults before applying the current file's overrides |
| [`schemas.py`](schemas.py) | Pydantic models: `ExperimentConfig` (root), `ModelConfig`, `DataConfig`, `TrainConfig`, `ComponentSpec` (`{type, params}`), `OptimizerParams`, `OptimizerSpec`, `ComponentConfigPaths`, `EMAConfig`, `CheckpointConfig`. Most use `ConfigDict(extra="allow")` to permit additional fields |
| [`errors.py`](errors.py) | `ConfigError` and `ConfigValidationIssue` — structured error types that format Pydantic validation failures into readable messages with YAML paths |

## Contracts

- `load_config(path)` returns a plain `dict`. `validate_config(raw)` turns it into a typed `ExperimentConfig` or raises `ConfigError`.
- The loader's `defaults` composition is *bottom-up then overlay*: each entry is loaded, deep-merged onto the running result, then the current file's keys are deep-merged on top.
- Cycles in `defaults` are detected via a stack and raise `ConfigLoadError`.

## How to extend

Add new validated fields to `ExperimentConfig` or its sub-models. Fields use Pydantic `Field` with validators for constraints. See the existing `lr > 0`, `decay in (0, 1)`, `max_epochs > 0` validators in [`schemas.py`](schemas.py) for examples.

When adding fields that the trainer entrypoints will actually read, declare them in the corresponding Pydantic model — don't rely on `extra="allow"`. See [docs/05_configuration.md § extra="allow" and undeclared fields](../../docs/05_configuration.md#extraallow-and-undeclared-fields) for why this matters.

## Cross-links

- Schema reference: [docs/05_configuration.md](../../docs/05_configuration.md)
- Hub: [docs/README.md](../../docs/README.md)

## Gotchas / invariants

- `ComponentSpec` uses `extra="allow"`, so unknown keys in `params` are preserved and forwarded to the component constructor via the `_filter_kwargs` machinery in [`utils/import_utils.py`](../utils/import_utils.py).
- The loader detects circular `defaults` references and raises `ConfigLoadError`.
- Relative paths in `defaults` entries are resolved against the directory of the config file that declares them, not the working directory.
- `TrainConfig` declares many fields that the entrypoints do not read; the entrypoints read several fields via `extra="allow"` that `TrainConfig` does not declare. The full mapping is in [docs/05_configuration.md § Where each field is consumed](../../docs/05_configuration.md#where-each-field-is-consumed).
