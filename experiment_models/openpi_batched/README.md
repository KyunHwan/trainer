# openpi_batched (model configs)

This folder contains model component configs for OpenPI-based batched-input training. For project-wide context, see [docs/README.md](../../docs/README.md).

## Purpose

Define the OpenPI model with a batched-input adapter, consumed by the openpi_batched imitation-learning trainer.

## Layout

| File | Component | Description |
|---|---|---|
| [`exp1/openpi_batched.yaml`](exp1/openpi_batched.yaml) | openpi_batched | Top-level model config wrapping the OpenPI architecture with batched-input support |

## Contracts

Consumed by:

- **Trainer**: [`openpi_batched_trainer`](../../experiment_training/components/trainer/imitation_learning/openpi_batched/openpi_batched_trainer.py)

Requires the `openpi_transformer_lib_patch.sh` shim to have been applied to the active venv — OpenPI relies on a patched version of the `transformers` library that's shipped inside the `policy_constructor` submodule. Apply with:

```bash
bash openpi_transformer_lib_patch.sh
```

after the virtualenv exists.

## Cross-links

- Trainer: [`../../experiment_training/components/trainer/imitation_learning/openpi_batched/`](../../experiment_training/components/trainer/imitation_learning/openpi_batched/)
- Transformer patch: [`../../openpi_transformer_lib_patch.sh`](../../openpi_transformer_lib_patch.sh)
- Hub: [docs/README.md](../../docs/README.md)
