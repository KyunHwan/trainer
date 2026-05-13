# naive_flow_matching_policy (model configs)

This folder contains model component configs for the basic flow-matching policy without variational bottleneck. For project-wide context, see [docs/README.md](../../docs/README.md).

## Purpose

Define the per-component architectures consumed by [`naive_flow_matching_policy_trainer`](../../experiment_training/components/trainer/imitation_learning/naive_flow_matching_policy/).

## Layout

| File | Component | Description |
|---|---|---|
| [`exp1/backbone.yaml`](exp1/backbone.yaml) | backbone | RadioV3 visual backbone over concatenated 3-camera input |
| [`exp1/da3.yaml`](exp1/da3.yaml) | da3 | Depth Anything v3 model (frozen, inference-only) |
| [`exp1/info_embedder.yaml`](exp1/info_embedder.yaml) | info_embedder | Transformer encoder fusing visual + depth + proprio |
| [`exp1/action_decoder.yaml`](exp1/action_decoder.yaml) | action_decoder | Flow-matching velocity decoder |
| [`exp1/proprio_projector.yaml`](exp1/proprio_projector.yaml) | proprio_projector | Projects proprio state into visual-token feature space |
| [`exp1/left_hand_extractor.yaml`](exp1/left_hand_extractor.yaml), [`right_hand_extractor.yaml`](exp1/right_hand_extractor.yaml) | hand extractors | Optional per-hand feature extractors |

## Contracts

Consumed by:

- **Trainer**: [`naive_flow_matching_policy_trainer`](../../experiment_training/components/trainer/imitation_learning/naive_flow_matching_policy/naive_flow_matching_policy_trainer.py)
- **Training config**: [`../../experiment_training/imitation_learning/naive_flow_matching_policy/exp1/naive_flow_matching_policy.yaml`](../../experiment_training/imitation_learning/naive_flow_matching_policy/)
- **Optimizer**: `schedule_free_radam` (no LR scheduler)
- **Loss**: none — the trainer's `forward` computes its own velocity-prediction MSE inline

## Cross-links

- Trainer: [`../../experiment_training/components/trainer/imitation_learning/naive_flow_matching_policy/`](../../experiment_training/components/trainer/imitation_learning/naive_flow_matching_policy/)
- Hub: [docs/README.md](../../docs/README.md)
