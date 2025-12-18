# configs/base/

## What this folder is

Reusable “base configs” that define defaults for an experiment.

In this repo, `configs/base/runtime.yaml` is the main base file.

## What a beginner should do here

- Treat `runtime.yaml` as the default wiring for *everything except the model*.
- Most experiments should include it via `defaults:` and then override just a few fields.

## Key files

- `configs/base/runtime.yaml`
  - `settings`: config/instantiation guardrails (imports/targets/prefixes)
  - default `data`, `io`, `loss`, `optimizer`, `trainer`
  - default `loggers` and `callbacks`

## How it connects to the pipeline

`runtime.yaml` contributes the training-side configuration, while `configs/models/*.yaml` contributes the model definition.

## Common edits / extension points

- Change default optimizer/loss/trainer across experiments: edit `runtime.yaml`.
- If you prefer per-experiment changes, override the relevant sections in `configs/experiments/*.yaml`.

## Links

- [Root config guide](../../README.md#config-system-what-yaml-keys-mean)
