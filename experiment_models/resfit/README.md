# resfit (model configs)

This folder contains model component configs for the ResFit residual-policy setup. For project-wide context, see [docs/README.md](../../docs/README.md).

## Purpose

Define the residual actor and Q-function consumed by [`resfit_trainer`](../../experiment_training/components/trainer/reinforcement_learning/resfit/). ResFit trains a small residual actor on top of a frozen base policy, with a learned Q-function for value estimation.

## Layout

| File | Component | Description |
|---|---|---|
| [`exp1/resfit_residual_actor.yaml`](exp1/resfit_residual_actor.yaml) | resfit_residual_actor | Small actor producing a residual added to the base-policy action. Has `online_update: true` so its weights are pushed to inference |
| [`exp1/resfit_q_function.yaml`](exp1/resfit_q_function.yaml) | resfit_q_function | Q-function trained alongside the actor. Has `online_update: false` (kept local) |

## Contracts

Consumed by:

- **Trainer**: [`resfit_trainer`](../../experiment_training/components/trainer/reinforcement_learning/resfit/resfit_trainer.py)
- **Training config**: [`../../experiment_training/reinforcement_learning/resfit/online_rl/resfit.yaml`](../../experiment_training/reinforcement_learning/resfit/online_rl/resfit.yaml)
- **Data**: `resfit_lerobot_dataset_factory` (LeRobot variant with `reward_horizon`)
- **Loss**: `l2_loss`
- **Trainer entrypoint**: [`trainer/online_trainer.py`](../../trainer/online_trainer.py) — this is an online-trainer experiment. See [docs/03_ray_online_training.md](../../docs/03_ray_online_training.md).

## Cross-links

- Trainer: [`../../experiment_training/components/trainer/reinforcement_learning/resfit/`](../../experiment_training/components/trainer/reinforcement_learning/resfit/)
- Hub: [docs/README.md](../../docs/README.md)
