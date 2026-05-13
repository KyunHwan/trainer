# Documentation index

*Estimated reading time: 5 minutes (this index only)*

This is the documentation hub for the `trainer` framework. Every document lives in a single `.md` file; this page tells you which one to open.

## Getting started

| If you are here to... | Read |
|---|---|
| Run your first training job on a single GPU in under 15 minutes | [01_getting_started.md](01_getting_started.md) |
| Understand the framework's mental model — registries, plugins, factories, protocols | [04_concepts.md](04_concepts.md) |
| Look up a YAML field, validator, or default | [05_configuration.md](05_configuration.md) |

## Reference

| Document | What's in it |
|---|---|
| [02_distributed_training.md](02_distributed_training.md) | What DDP does, how `torchrun` is wired up, the seed protocol, `SyncBatchNorm`, `find_unused_parameters`, rank-0 gating |
| [03_ray_online_training.md](03_ray_online_training.md) | Differences vs offline trainer, the `replay_buffer` and `policy_state_manager` actor contracts, weight broadcasting, batch concat semantics |
| [06_training_loop_walkthrough.md](06_training_loop_walkthrough.md) | Literate walkthrough of `train()` in `offline_trainer.py` stage by stage |
| [07_extending.md](07_extending.md) | Recipes: new dataset, new trainer, new loss, new optimizer |
| [08_checkpoints_and_resume.md](08_checkpoints_and_resume.md) | Disk layout, save/load semantics, partial loads, the `init`/`freeze` flags |
| [09_experiment_tracking.md](09_experiment_tracking.md) | wandb conventions, the metrics dict, rank-0-only logging |
| [12_glossary.md](12_glossary.md) | Terminology cheatsheet (DDP, autocast, MoE, OT, VFP, LeRobot, ...) |

## Operations

| Document | What's in it |
|---|---|
| [10_troubleshooting.md](10_troubleshooting.md) | Decision tree of failure modes |
| [11_testing.md](11_testing.md) | What the suite covers and how to run it |

## Per-folder reference

Each package directory has a `README.md` describing its purpose, layout, and contracts:

- Framework: [`trainer/`](../trainer/README.md), [`trainer/config/`](../trainer/config/README.md), [`trainer/modeling/`](../trainer/modeling/README.md), [`trainer/registry/`](../trainer/registry/README.md), [`trainer/templates/`](../trainer/templates/README.md), [`trainer/utils/`](../trainer/utils/README.md)
- Experiment plugins: [`experiment_training/`](../experiment_training/README.md), [`experiment_training/components/`](../experiment_training/components/README.md), [`components/dataloader/`](../experiment_training/components/dataloader/README.md), [`components/loss/`](../experiment_training/components/loss/README.md), [`components/optimizer/`](../experiment_training/components/optimizer/README.md), [`components/trainer/`](../experiment_training/components/trainer/README.md)
- Training configs: [`experiment_training/imitation_learning/`](../experiment_training/imitation_learning/README.md), [`experiment_training/reinforcement_learning/`](../experiment_training/reinforcement_learning/README.md)
- Model configs: [`experiment_models/`](../experiment_models/README.md) and one README per experiment subdirectory (e.g. [`vfp_single_expert/`](../experiment_models/vfp_single_expert/README.md))

## How the docs are organized

This documentation set is **breadth-first**. The numbered docs (`01`..`12`) are the primary reading order; the per-folder READMEs are reference material you jump to when you're already in a directory. The glossary is the safety net — if a term appears in any doc that a reader new to the codebase would not know, it is defined in [12_glossary.md](12_glossary.md).

When code is cited, you'll see `function_name()` linked to its file — for example, `_build_models()` in [`trainer/offline_trainer.py`](../trainer/offline_trainer.py). Open the file to verify — the docs are written so the code is the source of truth, not the other way around.
