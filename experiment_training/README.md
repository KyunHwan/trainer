# experiment_training

This folder contains concrete experiment implementations that plug into the [`trainer/`](../trainer/) framework via the registry system. For project-wide context, see [docs/README.md](../docs/README.md).

## Purpose

- Provide production training components (trainers, data loaders, losses, optimizers) for specific experiments
- Organize training YAML configs by experiment type and version
- Serve as a reference for implementing new experiments

## Layout

| Directory | Description |
|---|---|
| [`components/`](components/README.md) | Registered component implementations — the code that runs during training |
| [`imitation_learning/`](imitation_learning/README.md) | Training YAML configs for imitation-learning experiments |
| [`reinforcement_learning/`](reinforcement_learning/README.md) | Training YAML configs for reinforcement-learning experiments (resfit, dsrl_openpi) |

## Contracts

Modules in [`components/`](components/) are listed in the `plugins:` section of experiment YAML configs in [`imitation_learning/`](imitation_learning/) and [`reinforcement_learning/`](reinforcement_learning/). When the training entrypoint calls `load_plugins()`, these modules are imported, triggering their `@register` decorators and making components available to the registry system.

```yaml
plugins:
  - "experiment_training.components.dataloader.lerobot_data"
  - "experiment_training.components.trainer.imitation_learning.vfp_single_expert.vfp_single_expert_trainer"
  - "experiment_training.components.optimizer.adamw_cosine_decay"
  - "experiment_training.components.loss.sinkhorn_knopp"
```

## How to extend

Adding a new experiment is a four-step process:

1. Implement a trainer in `components/trainer/<algo>/<name>/` (and any custom dataset/loss/optimizer if needed).
2. Create model architecture configs in [`../experiment_models/<name>/`](../experiment_models/).
3. Create a training YAML config in `<algo>/<name>/exp1/`.
4. List the plugin module paths in the YAML.

Detailed recipes per component type: [docs/07_extending.md](../docs/07_extending.md).

## Cross-links

- Component framework: [docs/04_concepts.md](../docs/04_concepts.md)
- Hub: [docs/README.md](../docs/README.md)

## Gotchas / invariants

- Plugin module paths must be importable from the project root (e.g., `experiment_training.components.dataloader.lerobot_data`).
- Training YAML configs reference model architecture configs in [`../experiment_models/`](../experiment_models/) via relative paths that are resolved against the project root by `_build_models()`.
