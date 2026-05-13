# components

This folder contains registered component implementations for the training framework. For project-wide context, see [docs/README.md](../../docs/README.md).

## Purpose

- Implement concrete data loaders, losses, optimizers, and trainers
- Register each implementation under a string key for YAML-driven instantiation
- Provide production-ready components for imitation-learning and reinforcement-learning experiments

## Layout

| Subdirectory | Description | Registered keys |
|---|---|---|
| [`dataloader/`](dataloader/README.md) | Dataset factory implementations | `lerobot_dataset_factory`, `resfit_lerobot_dataset_factory`, `episodic_dataset_factory` |
| [`loss/`](loss/README.md) | Loss function factories | `l2_loss`, `sinkhorn_knopp` (+ standalone MoE gating utilities) |
| [`optimizer/`](optimizer/README.md) | Optimizer factory implementations | `adamw_warmup_cosine_decay`, `adamw_cosine_schedule`, `schedule_free_radam` |
| [`trainer/`](trainer/README.md) | Trainer implementations | See [`trainer/`](trainer/README.md) |

## Contracts

Each component file declares a class decorated with `@<REGISTRY>.register("key")` and implements the corresponding protocol from [`../../trainer/templates/`](../../trainer/templates/README.md). At runtime, the file's module path must appear in the YAML's `plugins:` list — only then does the registration decorator run.

## How to extend

See [docs/07_extending.md](../../docs/07_extending.md) for recipes per component type (dataset, trainer, loss, optimizer).

## Cross-links

- Concepts: [docs/04_concepts.md § Registries](../../docs/04_concepts.md#registries) and [§ Plugins](../../docs/04_concepts.md#plugins)
- Hub: [docs/README.md](../../docs/README.md)
