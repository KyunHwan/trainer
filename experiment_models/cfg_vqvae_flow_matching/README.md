# cfg_vqvae_flow_matching (model configs)

This folder contains model component configs for the CFG-VQVAE flow-matching policy. For project-wide context, see [docs/README.md](../../docs/README.md).

## Purpose

Define the per-component architectures consumed by the CFG-VQVAE K-OT trainer. The model combines a VQVAE codebook bottleneck with flow matching and supports classifier-free guidance at inference time.

## Layout

| File | Component | Description |
|---|---|---|
| [`exp1/backbone.yaml`](exp1/backbone.yaml) | backbone | Visual backbone |
| [`exp1/da3.yaml`](exp1/da3.yaml) | da3 | Depth Anything v3 model (frozen) |
| [`exp1/info_encoder.yaml`](exp1/info_encoder.yaml) | info_encoder | Fuses visual + proprio context |
| [`exp1/vqvae_posterior.yaml`](exp1/vqvae_posterior.yaml), [`vqvae_prior.yaml`](exp1/vqvae_prior.yaml), [`vqvae_codebook.yaml`](exp1/vqvae_codebook.yaml) | VQVAE | Codebook bottleneck with posterior/prior |
| [`exp1/action_decoder.yaml`](exp1/action_decoder.yaml) | action_decoder | Flow-matching velocity decoder, supports CFG dropout |
| [`exp1/proprio_projector.yaml`](exp1/proprio_projector.yaml) | proprio_projector | Projects proprio state into visual-token feature space |

## Contracts

Consumed by:

- **Trainer**: [`cfg_vqvae_flow_matching_trainer_kot`](../../experiment_training/components/trainer/imitation_learning/cfg_vqvae_flow_matching/cfg_vqvae_flow_matching_trainer_kot.py)
- **Training config**: [`../../experiment_training/imitation_learning/cfg_vqvae_flow_matching/exp1/cfg_vqvae_flow_matching.yaml`](../../experiment_training/imitation_learning/cfg_vqvae_flow_matching/)
- **Loss**: K-OT (typically Sinkhorn-based)

## Cross-links

- Trainer: [`../../experiment_training/components/trainer/imitation_learning/cfg_vqvae_flow_matching/`](../../experiment_training/components/trainer/imitation_learning/cfg_vqvae_flow_matching/)
- Hub: [docs/README.md](../../docs/README.md)
