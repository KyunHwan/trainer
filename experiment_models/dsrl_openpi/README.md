# dsrl_openpi (model configs)

This folder contains model component configs for DSRL (deep soft RL) layered on an OpenPI base policy. For project-wide context, see [docs/README.md](../../docs/README.md).

## Purpose

Define the OpenPI base model plus the DSRL actors and critics consumed by [`dsrl_openpi_trainer`](../../experiment_training/components/trainer/reinforcement_learning/dsrl_openpi/). The trainer operates in both the noise space and the latent space of the base policy.

## Layout

| File | Component | Description |
|---|---|---|
| [`exp1/openpi_model.yaml`](exp1/openpi_model.yaml) | openpi_model | Frozen OpenPI base policy |
| [`exp1/backbone.yaml`](exp1/backbone.yaml), [`da3.yaml`](exp1/da3.yaml) | Backbones | Visual backbone + Depth Anything v3 |
| [`exp1/noise_actor_img_encoder.yaml`](exp1/noise_actor_img_encoder.yaml), [`noise_actor.yaml`](exp1/noise_actor.yaml) | Noise actor | Image encoder + actor head over the noise space |
| [`exp1/noise_q_function_img_encoder.yaml`](exp1/noise_q_function_img_encoder.yaml), [`noise_q_function.yaml`](exp1/noise_q_function.yaml), [`noise_q_function_processor.yaml`](exp1/noise_q_function_processor.yaml) | Noise critic | Q-function over the noise space |
| [`exp1/noise_processor.yaml`](exp1/noise_processor.yaml) | noise_processor | Pre/post processing in the noise space |
| [`exp1/q_function_img_encoder.yaml`](exp1/q_function_img_encoder.yaml), [`q_function.yaml`](exp1/q_function.yaml), [`q_function_processor.yaml`](exp1/q_function_processor.yaml) | Action critic | Q-function over the action space |

## Contracts

Consumed by:

- **Trainer**: [`dsrl_openpi_trainer`](../../experiment_training/components/trainer/reinforcement_learning/dsrl_openpi/dsrl_openpi_trainer.py), with per-subnetwork helpers in [`utils/`](../../experiment_training/components/trainer/reinforcement_learning/dsrl_openpi/utils/) (action critic, noise-latent critic, noise-latent actor)
- **Training config**: [`../../experiment_training/reinforcement_learning/dsrl_openpi/exp1/dsrl_openpi.yaml`](../../experiment_training/reinforcement_learning/dsrl_openpi/exp1/dsrl_openpi.yaml)

Requires OpenPI weights and the `openpi_transformer_lib_patch.sh` shim. The base OpenPI policy is typically frozen via `component_build_args[openpi_model].freeze: true`.

## Cross-links

- Trainer: [`../../experiment_training/components/trainer/reinforcement_learning/dsrl_openpi/`](../../experiment_training/components/trainer/reinforcement_learning/dsrl_openpi/)
- Training configs: [`../../experiment_training/reinforcement_learning/`](../../experiment_training/reinforcement_learning/README.md)
- Hub: [docs/README.md](../../docs/README.md)
