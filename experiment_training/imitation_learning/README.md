# imitation_learning (training configs)

This folder contains training YAML configs for imitation-learning experiments. For project-wide context, see [docs/README.md](../../docs/README.md).

## Purpose

- Provide complete, ready-to-run training configurations
- Organize experiments by policy type and version
- Define all hyperparameters: plugins, model paths, data sources, optimizer schedules, loss functions, and training duration

## Layout

| Directory | Experiments | Algorithm | Prerequisites |
|---|---|---|---|
| [`vfp_single_expert/`](vfp_single_expert/) | `exp1/`, `exp2/` | Single-expert flow matching with `sinkhorn_knopp` OT loss. `exp1` uses 3 cameras (head, left, right); `exp2` adds Depth Anything v3 features as a 4th visual input | LeRobot dataset (default `joon001001/igris-b-pnp-v4.1`); `wandb login`; ~24 GB VRAM at the default batch size |
| [`cfg_vqvae_flow_matching/`](cfg_vqvae_flow_matching/) | `exp1/` | Classifier-free guidance VQVAE with flow matching and K-OT regularization | LeRobot dataset; `wandb login` |
| [`naive_flow_matching_policy/`](naive_flow_matching_policy/) | `exp1/`, `exp2/` | Basic flow matching policy without variational bottleneck | LeRobot dataset (default `joon001001/igris-b-pnp_v3.3.2`); schedule-free RAdam optimizer (no LR scheduler) |
| [`variational_flow_matching_policy/`](variational_flow_matching_policy/) | `exp1/`, `exp2/` | Variational flow matching with MoE action decoder, VQVAE codebook, posterior/prior. Requires `find_unused_parameters: true` due to MoE routing | LeRobot dataset; MoE-aware GPU (memory cost from `find_unused_parameters`) |
| [`mutual_information_estimator/`](mutual_information_estimator/) | `exp1/` | Mutual information estimation between state/action representations. Often used as a pretraining step for downstream policies | LeRobot dataset; runs without `loss` (the trainer computes losses internally from auto-encoder reconstructions) |

The most complete reference config — every field set, all five trainer components present — is [`vfp_single_expert/exp1/vfp_single_expert.yaml`](vfp_single_expert/exp1/vfp_single_expert.yaml). Start there when authoring new configs.

## How to extend

Adding a new experiment variant:

1. Copy an existing `expN/` directory to `expN+1/`.
2. Edit hyperparameters (learning rate, batch size, epochs, dataset `repo_id`, etc.).
3. Point `model.component_config_paths` to new architecture configs in [`../../experiment_models/`](../../experiment_models/) if architecture changes are needed.

Adding a new algorithm:

1. Implement the trainer in [`../components/trainer/imitation_learning/<name>/`](../components/trainer/imitation_learning/).
2. Create model configs in [`../../experiment_models/<name>/exp1/`](../../experiment_models/).
3. Create this YAML config at `<name>/exp1/<name>.yaml`.

See [docs/07_extending.md](../../docs/07_extending.md).

## Cross-links

- Configuration reference: [docs/05_configuration.md](../../docs/05_configuration.md)
- Concepts: [docs/04_concepts.md](../../docs/04_concepts.md)
- Hub: [docs/README.md](../../docs/README.md)

## Gotchas / invariants

- All paths in the config (model configs, save directories) are resolved relative to the project root.
- The `plugins` list must include all component modules needed by the config — missing plugins cause `KeyError` at registry lookup time. See [docs/10_troubleshooting.md § Registry / plugin failures](../../docs/10_troubleshooting.md#registry--plugin-failures).
- `save_dir` should be an absolute path or use `~` expansion. The training loop creates the directory and saves checkpoints as `epoch_<N>/<component>.pt`.
- `data.datamodule.params.task_name` becomes the wandb project name. Always include it.
