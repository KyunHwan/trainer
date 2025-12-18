# configs/models/

## What this folder is

Model architecture YAMLs for the vendored `policy_constructor/model_constructor` library.

These files define `model: ...` and are intentionally **training-agnostic**.

## What a beginner should do here

- Start by reading `sequential_mlp.yaml`.
- When authoring your own models, keep model YAMLs here and keep training components in `configs/base/` + `configs/experiments/`.

## Key files

- `configs/models/sequential_mlp.yaml`: minimal sequential model with a single `mlp` block.

## How it connects to the pipeline

- `offline_trainer` calls `build_model(config_path, registry=...)`, which uses the `model:` section from the resolved YAML.
- The resulting model exposes `model.inputs`, which is validated against `io.mapping.model_kwargs`.

## Common edits / extension points

- To use custom blocks/modules inside the model, register them into the model-constructor registry via YAML `imports:`.
  - See `policy_constructor/model_constructor/blocks/README.md` for the block/registry mechanics.

## Links

- [Root architecture](../../README.md#architecture-tour)
- [Model-constructor docs](../../policy_constructor/README.md)
