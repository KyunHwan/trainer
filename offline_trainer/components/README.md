# offline_trainer/components/

## What this folder is

Reference implementations of the **pluggable components** that `offline_trainer` wires together at runtime.

These are instantiated from YAML specs via registry keys (e.g. `_type_: data.random_regression`).

## What a beginner should do here

- Read this folder to understand what interfaces your custom components must implement.
- For your own experiments, prefer implementing new components under `extensions/` and registering them (instead of editing these files).

## Key files

- `offline_trainer/components/data.py`: `RandomRegressionDataModule` (yields batches).
- `offline_trainer/components/io.py`: `MappingIO` (extracts model inputs/targets from a batch).
- `offline_trainer/components/loss.py`: `TorchLoss` + `LossOutput`.
- `offline_trainer/components/optim.py`: `TorchOptimizerFactory`.
- `offline_trainer/components/sched.py`: `TorchSchedulerFactory` + `SchedulerBundle`.
- `offline_trainer/components/callbacks.py`: `ModelCheckpoint` callback.
- `offline_trainer/components/loggers.py`: `StdoutLogger`.
- `offline_trainer/components/metrics.py`: `NoOpMetric` (not registered by default).

## How it connects to the pipeline

In `offline_trainer/experiment.py`, these components are created by:

- `_instantiate(cfg[...], registry=..., settings=..., source_map=...)`

The built-in registry keys are registered in:

- `offline_trainer/registry/builtins.py`

## Common edits / extension points

Typical customizations (via a plugin module + YAML `imports:`):

- **Custom datamodule** (`data.*`): implement `setup()`, `train_dataloader()`, and optionally `val_dataloader()`.
- **Custom IO** (`io.*`): implement `split(batch, stage=...) -> (ArgsKwargs, targets, meta)`.
- **Custom loss** (`loss.*`): implement `__call__(outputs=..., targets=..., batch=..., stage=...) -> LossOutput`.
- **Custom optimizer class** (`optim_cls.*`): register a `torch.optim.Optimizer` subclass and continue using `optim_factory.torch`.
- **Custom optimizer factory** (`optim_factory.*`): implement `build(model, registry=...)`.

## Links

- [Root config guide](../../README.md#config-system-what-yaml-keys-mean)
- [Registry + plugins](../../README.md#registry--plugins-how-yaml-selects-components)
