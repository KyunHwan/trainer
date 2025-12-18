# configs/experiments/

## What this folder is

Run-level experiment configs. These are the YAML files you pass to `run_experiment()`.

They typically:

- include `../base/runtime.yaml` for training-side defaults
- include a model YAML under `../models/`
- override `run.*` (name/out_dir/etc.) and any components you want to change

## What a beginner should do here

- Start with `baseline.yaml`.
- Copy it to make your own experiment.

## Key files

- `configs/experiments/baseline.yaml`
  - includes runtime + `sequential_mlp.yaml`
  - sets `run.name`, `run.out_dir`, seed/device/precision
- `configs/experiments/custom_plugin.yaml`
  - adds `imports: [extensions.example_plugin]`
  - replaces `data` and `trainer` with plugin-provided keys
- `configs/experiments/sgd_step_lr.yaml`
  - extends `baseline.yaml`
  - overrides optimizer and enables a scheduler

## How it connects to the pipeline

These files are consumed by:

- `offline_trainer/experiment.py::run_experiment(config_path)`

## Common edits / extension points

- Use `_merge_: replace` + `_value_:` when you want to replace a value contributed by a default include (see `custom_plugin.yaml`).
- Add new custom components by listing your plugin module under `imports:`.

## Links

- [Root quickstart commands](../../README.md#quickstart)
- [Extensions guide](../../extensions/README.md)
