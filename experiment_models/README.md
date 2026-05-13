# experiment_models

This folder contains model architecture configurations for [policy_constructor](../policy_constructor/). Each YAML file defines a single model component (backbone, encoder, decoder, etc.) that is built into a `GraphModel` (PyTorch `nn.Module`) at training time. For project-wide context, see [docs/README.md](../docs/README.md).

## Purpose

- Define model architectures declaratively in YAML using the [policy_constructor config schema](../policy_constructor/model_constructor/config/)
- Organize model configs by experiment type and version
- Provide reusable component configs that can be mixed and matched across experiments

## Layout

```
experiment_models/
├── vfp_single_expert/             VFP single-expert flow matching
├── variational_flow_matching_policy/   VFP + MoE + VQVAE
├── naive_flow_matching_policy/    Basic flow matching policy
├── cfg_vqvae_flow_matching/       CFG-VQVAE flow matching
├── mutual_information_estimator/  MI estimation
├── resfit/                        ResFit residual policy components
├── openpi_batched/                OpenPI batched-input components
└── dsrl_openpi/                   DSRL with OpenPI base policy
```

Per-subfolder summaries:

| Subfolder | Consumed by | One-line description |
|---|---|---|
| [`vfp_single_expert/`](vfp_single_expert/README.md) | `vfp_single_expert_trainer` (and `_depth` variant) | 3-camera flow-matching policy with transformer info embedder + causal transformer action decoder |
| [`variational_flow_matching_policy/`](variational_flow_matching_policy/README.md) | `variational_flow_matching_policy_trainer` | VFP with MoE action decoder, VQVAE posterior/prior, gating network |
| [`naive_flow_matching_policy/`](naive_flow_matching_policy/README.md) | `naive_flow_matching_policy_trainer` | Basic flow-matching policy without variational bottleneck |
| [`cfg_vqvae_flow_matching/`](cfg_vqvae_flow_matching/README.md) | `cfg_vqvae_flow_matching_trainer_kot` | CFG-VQVAE flow-matching with K-OT regularization |
| [`mutual_information_estimator/`](mutual_information_estimator/README.md) | `mutual_information_estimator_trainer` | State/action autoencoders for MI estimation |
| [`resfit/`](resfit/README.md) | `resfit_trainer` (online RL) | Residual actor + Q-function over a frozen base policy |
| [`openpi_batched/`](openpi_batched/README.md) | OpenPI batched trainer | OpenPI model with batched-input adapter |
| [`dsrl_openpi/`](dsrl_openpi/README.md) | `dsrl_openpi_trainer` | OpenPI base + DSRL critics/actors over noise and latent spaces |

## Contracts

Training YAML configs reference these files via `model.component_config_paths`:

```yaml
model:
  component_config_paths:
    head_backbone: "experiment_models/vfp_single_expert/exp1/head_backbone.yaml"
    info_embedder: "experiment_models/vfp_single_expert/exp1/info_embedder.yaml"
    action_decoder: "experiment_models/vfp_single_expert/exp1/action_decoder.yaml"
```

At training time, [`PolicyConstructorModelFactory`](../trainer/modeling/factories.py) calls `model_constructor.build_model(config_path)` for each entry, producing named `nn.Module` instances stored in an `nn.ModuleDict`.

Each YAML in this tree follows the policy_constructor schema (declarative `params:` + `model.graph` with `modules`, `nodes`, `inputs`, `outputs`). The schema is defined in [`policy_constructor/model_constructor/config/`](../policy_constructor/model_constructor/config/) and documented in [`policy_constructor/README.md`](../policy_constructor/README.md). **This repository does not re-document that schema** — link, don't duplicate.

## How to extend

Two scenarios:

1. **A new variant of an existing experiment.** Copy `vfp_single_expert/exp1/` to `vfp_single_expert/exp3/`, adjust hyperparameters (layer counts, hidden dims, attention heads), and reference the new paths from a training YAML in [`../experiment_training/imitation_learning/`](../experiment_training/imitation_learning/).
2. **A new experiment type.** Create a new top-level subfolder, define one YAML per model component using the policy_constructor schema, then create the matching trainer in [`../experiment_training/components/trainer/`](../experiment_training/components/trainer/) and a training config in [`../experiment_training/imitation_learning/`](../experiment_training/imitation_learning/) (or `reinforcement_learning/`).

## Cross-links

- policy_constructor schema: [`policy_constructor/README.md`](../policy_constructor/README.md) and [`policy_constructor/model_constructor/config/`](../policy_constructor/model_constructor/config/)
- Model factory: [`trainer/modeling/`](../trainer/modeling/README.md)
- Hub: [docs/README.md](../docs/README.md)

## Gotchas / invariants

- Config paths are relative to the project root and resolved at training time by [`_build_models()`](../trainer/offline_trainer.py). Absolute paths are also supported.
- Each YAML file defines a single model component — the trainer is responsible for orchestrating the interaction between components.
- Model configs use the policy_constructor registry to reference block types (e.g., `vfp_single_action_decoder`, `radiov3`). These blocks must be registered before model construction (the submodule does this at import time).
