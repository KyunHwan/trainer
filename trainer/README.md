# trainer

This folder contains the core training framework. For project-wide context, see [docs/README.md](../docs/README.md).

## Purpose

- Load and validate YAML experiment configs via Pydantic schemas
- Discover and register custom components through a plugin system
- Build models from [policy_constructor](../policy_constructor/) configs via a thin adapter
- Run distributed training loops with PyTorch DDP or Ray Train
- Checkpoint models/optimizers and log metrics to Weights & Biases

## Layout

| File / dir | Description |
|---|---|
| [`offline_trainer.py`](offline_trainer.py) | DDP-based offline training entrypoint. CLI flag: `--train_config`. Defines `train(config_path)` and the epoch-based training loop |
| [`online_trainer.py`](online_trainer.py) | Ray Train online/offline hybrid entrypoint. Defines `train_func(config_path)` for use as a Ray worker function |
| [`config/`](config/README.md) | YAML loader with `defaults` composition, Pydantic schema validation, and structured error reporting |
| [`modeling/`](modeling/README.md) | `PolicyConstructorModelFactory` — adapter that calls `model_constructor.build_model()` per component |
| [`registry/`](registry/README.md) | Generic typed registry (`Registry[T]`) and the four global registries (trainer, dataset, optimizer, loss). Plugin loader for dynamic imports |
| [`templates/`](templates/README.md) | Python `Protocol` definitions for `Trainer`, `DatasetFactory`, `LossFactory`, `OptimizerFactory` — the contracts that all registered components must satisfy |
| [`utils/`](utils/README.md) | `tree_map` for nested structures, `move_to_device`/`cast_dtype` for tensors, `set_global_seed`/`seed_worker` for reproducibility, `select` for dict/tuple access, `instantiate` for filtered kwargs construction |

## Contracts

Both entrypoints follow the same pipeline:

```text
YAML → load_config → validate_config → load_plugins → _build_trainer → loop
```

The loop calls `trainer.train_step(data=data, stats=stats_gpu)` once per batch under `torch.autocast(dtype=torch.bfloat16)`. Normalization is **not** applied by the framework — it is the trainer's responsibility (see [docs/04_concepts.md § The stats dict and where normalization lives](../docs/04_concepts.md#the-stats-dict-and-where-normalization-lives)).

## How to extend

See [docs/07_extending.md](../docs/07_extending.md) for recipes covering all four extension points (trainer, dataset, loss, optimizer).

## Cross-links

- Concepts: [docs/04_concepts.md](../docs/04_concepts.md)
- Loop walkthrough: [docs/06_training_loop_walkthrough.md](../docs/06_training_loop_walkthrough.md)
- Distributed training: [docs/02_distributed_training.md](../docs/02_distributed_training.md)
- Ray online training: [docs/03_ray_online_training.md](../docs/03_ray_online_training.md)
- Hub: [docs/README.md](../docs/README.md)

## Gotchas / invariants

- **DDP wrapping**: Models are wrapped with `DistributedDataParallel` only when `world_size > 1` and the model is not frozen. Frozen models are moved to device but not wrapped. Checkpoints always save unwrapped `.module` state dicts. Defined in `_build_models()` in [`offline_trainer.py`](offline_trainer.py).
- **Rank-0-only operations**: wandb logging, checkpoint saving, and stats persistence are gated on `rank == 0`. All ranks must hit barriers together (see [docs/02_distributed_training.md § Rank-0-only side effects](../docs/02_distributed_training.md#rank-0-only-side-effects)).
- **Seed management**: Base seed is shared across ranks for synchronized weight init. After model construction, seed is offset by rank for independent dropout/augmentation randomness. See `train()` body in [`offline_trainer.py`](offline_trainer.py).
- **SyncBatchNorm**: Automatically applied when BatchNorm layers are detected in DDP mode, before moving to device. See `_build_models()` in [`offline_trainer.py`](offline_trainer.py).
- **Mixed precision**: Forward passes run under `torch.autocast(dtype=torch.bfloat16)`. Data is cast to `float32` before entering the autocast region.
