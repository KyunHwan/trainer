# mutual_information_estimator (model configs)

This folder contains model component configs for the mutual-information estimator. For project-wide context, see [docs/README.md](../../docs/README.md).

## Purpose

Define the per-component architectures consumed by [`mutual_information_estimator_trainer`](../../experiment_training/components/trainer/imitation_learning/mutual_information_estimator/). The model trains autoencoders over states (proprio + 128×128 head camera) and actions to produce shared low-entropy representations whose mutual information can be estimated.

## Layout

| File | Component | Description |
|---|---|---|
| [`exp1/action_encoder.yaml`](exp1/action_encoder.yaml) | action_encoder | Action → embedding encoder |
| [`exp1/action_decoder.yaml`](exp1/action_decoder.yaml) | action_decoder | Embedding → action reconstruction |
| [`exp1/state_resnet34_encoder.yaml`](exp1/state_resnet34_encoder.yaml) | state_encoder | ResNet-34-based state (image + proprio) encoder |
| [`exp1/state_resnet34_decoder.yaml`](exp1/state_resnet34_decoder.yaml) | state_decoder | State decoder (image + proprio reconstruction) |

## Contracts

Consumed by:

- **Trainer**: [`mutual_information_estimator_trainer`](../../experiment_training/components/trainer/imitation_learning/mutual_information_estimator/mutual_information_estimator_trainer.py)
- **Training config**: [`../../experiment_training/imitation_learning/mutual_information_estimator/exp1/mutual_information_estimator.yaml`](../../experiment_training/imitation_learning/mutual_information_estimator/)
- **Loss**: none — the trainer computes reconstruction MSE inline and adds an L2 regularizer on the embeddings

This experiment is typically used as a pretraining step; the resulting encoders can be loaded into downstream policy trainers via `train.load_dir` with the appropriate `.pt` files.

## Cross-links

- Trainer: [`../../experiment_training/components/trainer/imitation_learning/mutual_information_estimator/`](../../experiment_training/components/trainer/imitation_learning/mutual_information_estimator/)
- Hub: [docs/README.md](../../docs/README.md)
