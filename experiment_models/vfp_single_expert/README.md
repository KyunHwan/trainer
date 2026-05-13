# vfp_single_expert (model configs)

This folder contains model component configs for the VFP single-expert flow-matching policy. For project-wide context, see [docs/README.md](../../docs/README.md).

## Purpose

Define the per-component PyTorch model architectures consumed by the [`vfp_single_expert_trainer`](../../experiment_training/components/trainer/imitation_learning/vfp_single_expert/) trainer.

## Layout

| Path | Component | Description |
|---|---|---|
| [`exp1/`](exp1/) | Baseline | 3-camera VFP single-expert |
| [`exp1/head_backbone.yaml`](exp1/head_backbone.yaml) | head_backbone | RadioV3 backbone over the head camera |
| [`exp1/left_backbone.yaml`](exp1/left_backbone.yaml) | left_backbone | RadioV3 backbone over the left camera |
| [`exp1/right_backbone.yaml`](exp1/right_backbone.yaml) | right_backbone | RadioV3 backbone over the right camera |
| [`exp1/info_embedder.yaml`](exp1/info_embedder.yaml) | info_embedder | Transformer encoder fusing proprio + 3 visual streams + semantic tokens |
| [`exp1/action_decoder.yaml`](exp1/action_decoder.yaml) | action_decoder | Causal transformer decoder predicting flow-matching velocity |
| [`exp1/vae_posterior.yaml`](exp1/vae_posterior.yaml), [`vae_prior.yaml`](exp1/vae_prior.yaml) | Optional VAE | Posterior/prior transformers (used by ablation variants) |
| [`exp1/da3.yaml`](exp1/da3.yaml) | da3 | Depth Anything v3 model — used by the `_depth` trainer variant |
| [`exp2/`](exp2/) | Depth variant | Same as exp1 with `multimodal_bridge.yaml` and deeper `action_decoder.yaml` |

## Contracts

Consumed by:

- **Trainer**: [`vfp_single_expert_trainer`](../../experiment_training/components/trainer/imitation_learning/vfp_single_expert/vfp_single_expert_trainer.py) (exp1) and [`vfp_single_expert_trainer_depth`](../../experiment_training/components/trainer/imitation_learning/vfp_single_expert/vfp_single_expert_trainer_depth.py) (exp2)
- **Training config**: [`../../experiment_training/imitation_learning/vfp_single_expert/exp1/vfp_single_expert.yaml`](../../experiment_training/imitation_learning/vfp_single_expert/exp1/vfp_single_expert.yaml) and `exp2/vfp_single_expert_depth.yaml`
- **Dataset**: `lerobot_dataset_factory` over a LeRobot dataset
- **Loss**: `sinkhorn_knopp` (Sinkhorn-Knopp OT loss with state weighting)

## Cross-links

- Trainer: [`../../experiment_training/components/trainer/imitation_learning/vfp_single_expert/`](../../experiment_training/components/trainer/imitation_learning/vfp_single_expert/)
- Training configs: [`../../experiment_training/imitation_learning/vfp_single_expert/`](../../experiment_training/imitation_learning/vfp_single_expert/)
- policy_constructor schema: [`../../policy_constructor/README.md`](../../policy_constructor/README.md)
- Hub: [docs/README.md](../../docs/README.md)
