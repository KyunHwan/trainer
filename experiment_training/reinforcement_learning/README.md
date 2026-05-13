# reinforcement_learning (training configs)

This folder contains training YAML configs for reinforcement-learning experiments. For project-wide context, see [docs/README.md](../../docs/README.md).

## Purpose

- Provide ready-to-run RL training configurations
- Organize experiments by algorithm and version

## Layout

| Directory | Experiments | Algorithm | Prerequisites |
|---|---|---|---|
| [`resfit/online_rl/`](resfit/online_rl/) | `resfit.yaml` | ResFit residual actor + Q-function. Designed for the Ray-based online trainer ([`trainer/online_trainer.py`](../../trainer/online_trainer.py)). Residual actor has `online_update: true` so weights are pushed to inference; Q-function has `online_update: false` (trained but kept local) | A running Ray cluster with `replay_buffer` and `policy_state_manager` named actors. See [docs/03_ray_online_training.md](../../docs/03_ray_online_training.md) |
| [`dsrl_openpi/exp1/`](dsrl_openpi/exp1/) | `dsrl_openpi.yaml` | DSRL (deep soft RL) layered on an OpenPI base policy. Multiple critic components (action critic, noise/latent critic, noise/latent actor) plus a frozen OpenPI backbone | OpenPI weights; the `openpi_transformer_lib_patch.sh` shim must have been applied to the active venv |

The trainer implementations live in [`../components/trainer/reinforcement_learning/`](../components/trainer/reinforcement_learning/) (separately for `resfit/` and `dsrl_openpi/`, each with their own `utils/` for per-subnetwork training helpers).

## Contracts

- The `resfit` config is intended for `trainer/online_trainer.py train_func(...)`. It will *not* work with the offline trainer because the online trainer is the only one that polls the `replay_buffer` actor and pushes weights via `policy_state_manager`.
- The `dsrl_openpi` config can run offline (it has no `online_update` flags). Use `python trainer/offline_trainer.py --train_config ...` or `torchrun ...`.

## How to extend

To add a new RL algorithm:

1. Implement the trainer in [`../components/trainer/reinforcement_learning/<name>/`](../components/trainer/reinforcement_learning/).
2. Create model configs in [`../../experiment_models/<name>/`](../../experiment_models/).
3. Create this YAML config at `<algo>/<name>/<name>.yaml`.
4. Decide which components need `online_update: true` if the trainer is meant for the online path.

See [docs/07_extending.md](../../docs/07_extending.md).

## Cross-links

- Ray online training: [docs/03_ray_online_training.md](../../docs/03_ray_online_training.md)
- Concepts: [docs/04_concepts.md](../../docs/04_concepts.md)
- Hub: [docs/README.md](../../docs/README.md)
