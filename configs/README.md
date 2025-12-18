# configs/

## What this folder is

Experiment configuration for this repo.

- `offline_trainer` reads a single “experiment YAML” file.
- That YAML is **resolved** (includes/merge/interpolation) by the vendored `policy_constructor/model_constructor` config system.

## What a beginner should do here

- Start by running `configs/experiments/baseline.yaml`.
- Create new experiments by composing `base/` + `models/` and overriding only what you need.

## Key folders

- `configs/base/`: reusable defaults (runtime settings + default components).
- `configs/models/`: model-constructor YAMLs (architecture only).
- `configs/experiments/`: run-level experiment configs that compose base + model.

## How it connects to the pipeline

`offline_trainer/experiment.py` calls `resolve_config(path)` on the YAML you pass in.

Resolution features used in this repo include:

- `defaults:` includes (see `configs/experiments/*.yaml`)
- `${...}` interpolation (see `${run.name}` and `${run.out_dir}`)
- list merge directives (`_merge_` / `_value_`) (see `custom_plugin.yaml`)

For the full resolver semantics, see:
- `policy_constructor/model_constructor/config/README.md`

## Common edits / extension points

- Add a new experiment: copy an existing file in `configs/experiments/`.
- Swap trainer/data/loss/etc.: change the `_type_` keys in the experiment YAML.
- Add custom components: implement them in `extensions/` and add them to the experiment YAML `imports:`.

## Links

- [Root quickstart](../README.md#quickstart)
- [Root config guide](../README.md#config-system-what-yaml-keys-mean)
