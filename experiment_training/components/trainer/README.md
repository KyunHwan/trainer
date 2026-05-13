# trainer (components)

This folder contains training loop implementations registered in `TRAINER_REGISTRY`. For project-wide context, see [docs/README.md](../../../docs/README.md).

## Purpose

- Implement the `Trainer` protocol for specific policy architectures
- Encapsulate the full training step: forward → loss → backward → clip → step
- Manage multi-model orchestration (backbones, encoders, decoders) within a single `train_step()`

## Layout

```
trainer/
├── imitation_learning/                          IL trainer implementations
│   ├── vfp_single_expert/                       VFP single-expert trainers
│   ├── cfg_vqvae_flow_matching/                 CFG-VQVAE flow matching trainer
│   ├── naive_flow_matching_policy/              Basic flow matching policy trainer
│   ├── variational_flow_matching_policy/        Variational flow matching with MoE
│   ├── mutual_information_estimator/            MI estimation trainer
│   └── openpi_batched/                          OpenPI batched-input trainer
└── reinforcement_learning/                      RL trainer implementations
    ├── dsrl_openpi/                             DSRL with OpenPI base policy
    └── resfit/                                  ResFit residual-policy trainer
```

Per-algorithm summaries live in [`imitation_learning/README.md`](imitation_learning/README.md) and in this folder's `reinforcement_learning/` subdirectories.

## Contracts

Each trainer implements the `Trainer` protocol. The canonical call site is:

```python
loss_dict = trainer.train_step(data=data, stats=stats_gpu)
```

passed under `torch.autocast(dtype=torch.bfloat16)` once per batch. The trainer is expected to zero grads, forward, backward, optionally clip, step optimizers, and return a flat dict of metric values.

The trainer's `__init__` is called with `models: nn.ModuleDict`, `optimizers: dict`, `loss: nn.Module`, `device` (plus any `train.trainer.params` from YAML, filtered by signature).

> **Note**: some existing trainer files declare `train_step(self, data, epoch, total_epochs, iterations)` instead of `(data, stats)`. The framework calls `train_step(data=..., stats=...)`; trainers that don't accept those kwargs will fail with `TypeError`. The canonical signature is `(data, stats)`. See [docs/04_concepts.md § The train_step contract](../../../docs/04_concepts.md#the-train_step-contract).

## How to extend

See [docs/07_extending.md § Recipe: a new trainer](../../../docs/07_extending.md#recipe-a-new-trainer).

Recommended file layout for a new trainer:

```
imitation_learning/<name>/
├── __init__.py                   # import the trainer module so the @register fires
└── <name>_trainer.py             # the @TRAINER_REGISTRY.register-decorated class
```

Then add `experiment_training.components.trainer.imitation_learning.<name>.<name>_trainer` to your YAML's `plugins:` list.

## Cross-links

- Trainer contract: [docs/04_concepts.md § The train_step contract](../../../docs/04_concepts.md#the-train_step-contract)
- Recipe: [docs/07_extending.md § Recipe: a new trainer](../../../docs/07_extending.md#recipe-a-new-trainer)
- Loop walkthrough: [docs/06_training_loop_walkthrough.md](../../../docs/06_training_loop_walkthrough.md)
- Hub: [docs/README.md](../../../docs/README.md)

## Gotchas / invariants

- The `train_step()` return dict is logged to wandb on rank 0. Tensor values must be scalar; non-scalar tensors will raise when `.detach().item()` is called.
- Trainers manage their own gradient clipping and learning-rate logging. The framework does not call `clip_grad_norm_` for you.
- Multi-model trainers can selectively freeze/unfreeze models per-component via `model.component_build_args` and skip an optimizer for any component by omitting it from `model.component_optims`.
