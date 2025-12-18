# offline_trainer/config/

## What this folder is

Config validation and error-formatting helpers for `offline_trainer`.

`offline_trainer` intentionally validates configs in two phases:

- **preflight**: validate the YAML structure before building the model
- **post-model**: validate registry keys and IO mappings after the model is constructed

## What a beginner should do here

- If you hit a `ConfigError`, this folder is where the checks are implemented.
- If you are authoring YAML, read the validation rules to understand what shapes/keys are expected.

## Key files

- `offline_trainer/config/validate.py`
  - `validate_preflight(cfg, source_map=...) -> RunConfig`
  - `validate_post_model(cfg, registry=..., source_map=..., model=...) -> None`
- `offline_trainer/config/errors.py`
  - `enrich_config_error(...)`: attaches YAML `loc=...` when possible
  - `format_error_with_context(...)`: prints a small YAML snippet around the error location

## How it connects to the pipeline

`offline_trainer/experiment.py` calls:

- `validate_preflight()` immediately after `resolve_config()`
- `validate_post_model()` after `build_model()` and registry construction

## Common edits / extension points

- Add new config invariants: extend `validate_preflight()` and/or `validate_post_model()`.
- Keep validation error messages actionable: include a `config_path` so users know what to fix.

## Links

- [Root debugging section](../../README.md#debugging-config-errors)
