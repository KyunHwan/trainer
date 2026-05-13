# imitation_learning (trainer implementations)

This folder contains imitation-learning trainer implementations. For project-wide context, see [docs/README.md](../../../../docs/README.md).

## Purpose

- Implement the training step for various imitation-learning policy architectures
- Handle multi-camera visual processing, flow matching, and variational methods
- Manage per-component forward passes across backbones, encoders, and decoders

## Layout

| Directory | Registry key | Policy type |
|-----------|-------------|-------------|
| [`vfp_single_expert/`](vfp_single_expert/) | `vfp_single_expert_trainer` | Single-expert flow matching with 3 cameras (head, left, right). Beta(1.0, 1.5) time sampling, random camera dropout (15%), transformer info embedder + action decoder. Velocity loss (L2) |
| [`vfp_single_expert/`](vfp_single_expert/) | `vfp_single_expert_trainer_depth` | Same as above with Depth Anything v3 features as a 4th visual input. L1 velocity loss |
| [`cfg_vqvae_flow_matching/`](cfg_vqvae_flow_matching/) | `cfg_vqvae_flow_matching_trainer_kot` | Classifier-free guidance VQVAE with flow matching and K-OT regularization |
| [`naive_flow_matching_policy/`](naive_flow_matching_policy/) | `naive_flow_matching_policy_trainer` | Basic flow matching policy without variational bottleneck |
| [`variational_flow_matching_policy/`](variational_flow_matching_policy/) | `variational_flow_matching_policy_trainer` | Variational flow matching with MoE action decoder, VQVAE codebook, posterior/prior |
| [`mutual_information_estimator/`](mutual_information_estimator/) | `mutual_information_estimator_trainer` | Mutual information estimation between state/action representations |
| [`openpi_batched/`](openpi_batched/) | *(see file)* | OpenPI batched-input trainer (requires the `openpi_transformer_lib_patch.sh` shim) |

## Common training step pattern

All trainers follow a shared structure:

1. **Extract features** — pass camera images through backbone models
2. **Encode conditioning** — combine visual + proprioceptive features via info embedder
3. **Flow matching** — sample noise and time, interpolate between noise and target action, compute target velocity
4. **Decode** — predict velocity via action decoder
5. **Loss** — compute velocity prediction loss (and any auxiliary losses)
6. **Update** — zero grad → backward → clip gradients → optimizer step

## Contracts

Trainers receive `models` (nn.ModuleDict of named model components), `optimizers` (dict of per-component optimizers), `loss` (nn.Module or `None`), and `device`. The training loop calls `train_step(data=data, stats=stats_gpu)` and logs the returned metric dict to wandb on rank 0.

Some trainers in this folder declare older `(data, epoch, total_epochs, iterations)` signatures for `train_step`; the entrypoint calls with `(data, stats)` only. See [docs/04_concepts.md § The train_step contract](../../../../docs/04_concepts.md#the-train_step-contract).

## Cross-links

- Recipe: [docs/07_extending.md § Recipe: a new trainer](../../../../docs/07_extending.md#recipe-a-new-trainer)
- Trainer contract: [docs/04_concepts.md § The train_step contract](../../../../docs/04_concepts.md#the-train_step-contract)
- Hub: [docs/README.md](../../../../docs/README.md)
