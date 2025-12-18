# offline_trainer/deps/

## What this folder is

A small dependency shim that lets this repo work in two modes:

1. Use an installed `model_constructor` package (if available)
2. Fall back to the vendored submodule at `policy_constructor/model_constructor`

This keeps the training code independent from how the model-constructor is provided.

## What a beginner should do here

Usually nothing. This folder exists so the rest of the codebase can import a stable surface:

- `offline_trainer.deps.model_constructor`

## Key files

- `offline_trainer/deps/model_constructor.py`: re-exports `Registry`, `build_model`, `resolve_config`, `instantiate_value`, and error types.

## How it connects to the pipeline

`offline_trainer/experiment.py` imports the model-constructor API only from this shim.

## Common edits / extension points

- Only change this if you are deliberately swapping out the model-constructor dependency.

## Links

- [Root architecture](../../README.md#architecture-tour)
