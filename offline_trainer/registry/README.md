# offline_trainer/registry/

## What this folder is

The registry wiring for `offline_trainer`.

A **registry** maps string keys (used in YAML) to Python callables/classes (instantiated at runtime).

## What a beginner should do here

- If you only want to *use* the repo, you typically do not edit this folder.
- If you are adding a new built-in component type, this folder is where you register it.
- For project-specific components, prefer writing a plugin under `extensions/` and importing it via YAML `imports:`.

## Key files

- `offline_trainer/registry/builder.py`: `build_registry()` creates a `Registry` and registers:
  - model_constructor built-ins + blocks
  - offline_trainer built-ins
- `offline_trainer/registry/builtins.py`: `register_offline_trainer_builtins(registry)` registers the built-in keys.

## How it connects to the pipeline

`offline_trainer/experiment.py` constructs a registry once per run and uses it to:

- build the model (model-constructor modules/blocks)
- instantiate trainer/data/IO/loss/optimizer factory/scheduler factory/callbacks/loggers

## Common edits / extension points

- Add built-ins: update `register_offline_trainer_builtins()`.
- Add custom components without editing core code:
  - write a module with `register(registry)`
  - list it under YAML `imports:`

Important: registry keys must be globally unique. Duplicates will surface as a registry/import error during model build.

## Links

- [Root registry overview](../../README.md#registry--plugins-how-yaml-selects-components)
- [Extensions guide](../../extensions/README.md)
