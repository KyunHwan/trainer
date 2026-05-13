# variational_flow_matching_policy (model configs)

This folder contains model component configs for the variational flow-matching policy with MoE. For project-wide context, see [docs/README.md](../../docs/README.md).

## Purpose

Define the per-component architectures consumed by [`variational_flow_matching_policy_trainer`](../../experiment_training/components/trainer/imitation_learning/variational_flow_matching_policy/).

## Layout

| File | Component | Description |
|---|---|---|
| [`exp1/backbone.yaml`](exp1/backbone.yaml) | backbone | Shared visual backbone (forward over concatenated head/left/right) |
| [`exp1/da3.yaml`](exp1/da3.yaml) | da3 | Depth Anything v3 model (no_grad inference) |
| [`exp1/info_embedder.yaml`](exp1/info_embedder.yaml) | info_embedder | Transformer encoder fusing visual + depth + proprio |
| [`exp1/vqvae_posterior.yaml`](exp1/vqvae_posterior.yaml), [`vqvae_prior.yaml`](exp1/vqvae_prior.yaml), [`vqvae_codebook.yaml`](exp1/vqvae_codebook.yaml) | VQVAE | Variational bottleneck (posterior/prior/codebook) |
| [`exp1/gate.yaml`](exp1/gate.yaml) | gate | Routing network for the MoE action decoder |
| [`exp1/moe_action_decoder.yaml`](exp1/moe_action_decoder.yaml) | moe_action_decoder | Multi-expert flow-matching velocity decoder |
| [`exp1/proprio_projector.yaml`](exp1/proprio_projector.yaml) | proprio_projector | Projects proprio state into visual-token feature space |
| [`exp1/left_hand_extractor.yaml`](exp1/left_hand_extractor.yaml), [`right_hand_extractor.yaml`](exp1/right_hand_extractor.yaml) | hand extractors | Optional per-hand feature extractors |

## Contracts

Requires `find_unused_parameters: true` because MoE routing produces input-dependent parameter usage. See [docs/02_distributed_training.md § find_unused_parameters](../../docs/02_distributed_training.md#find_unused_parameters).

Consumed by:

- **Trainer**: [`variational_flow_matching_policy_trainer`](../../experiment_training/components/trainer/imitation_learning/variational_flow_matching_policy/variational_flow_matching_policy_trainer.py)
- **Training config**: [`../../experiment_training/imitation_learning/variational_flow_matching_policy/exp1/variational_flow_matching_policy.yaml`](../../experiment_training/imitation_learning/variational_flow_matching_policy/)
- **Loss**: `sinkhorn_knopp`

## Cross-links

- Trainer: [`../../experiment_training/components/trainer/imitation_learning/variational_flow_matching_policy/`](../../experiment_training/components/trainer/imitation_learning/variational_flow_matching_policy/)
- Hub: [docs/README.md](../../docs/README.md)
