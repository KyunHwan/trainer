# offline_trainer/engine/

## What this folder is

The training runtime:

- the default training loop (`DefaultTrainer`)
- device/precision utilities (`SingleDeviceAccelerator`)
- checkpoint capture/restore
- state containers and small helpers

## What a beginner should do here

- If you want to understand “what happens during training”, start with `trainer.py`.
- If you want a different loop (multi-device, custom logging, different step semantics), implement a custom trainer and register it via a plugin.

## Key files

- `offline_trainer/engine/trainer.py`: `DefaultTrainer.fit(...)` (train/val loop, grad accumulation, scheduler stepping, logging, resume).
- `offline_trainer/engine/accelerator.py`: `SingleDeviceAccelerator` (device + mixed precision + grad scaler).
- `offline_trainer/engine/checkpoint.py`: checkpoint format + save/load/restore helpers.
- `offline_trainer/engine/state.py`: `TrainState` and `RunContext` (objects passed to callbacks/loggers).
- `offline_trainer/engine/tree.py`: `move_to_device()` and `to_float()` utilities.
- `offline_trainer/engine/seed.py`: deterministic seeding + RNG state capture/restore.

## How it connects to the pipeline

The registry key `trainer.default` maps to `DefaultTrainer` and is selected via YAML `trainer: {_type_: trainer.default, ...}`.

Checkpointing is typically enabled via the callback key `cb.model_checkpoint`.

## Common edits / extension points

- **Custom trainer** (`trainer.*`): implement a `fit(...)` method that accepts the same keyword args as `DefaultTrainer.fit`.
- **Callbacks** (`cb.*`): implement hooks like `on_fit_start(ctx)` and/or `on_train_step_end(ctx)`.
- **Loggers** (`log.*`): implement `log_metrics(metrics, step=..., stage=...)`.

## Links

- [Root architecture](../../README.md#architecture-tour)
- [Root config guide](../../README.md#config-system-what-yaml-keys-mean)
