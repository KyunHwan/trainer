# tests/

## What this folder is

Minimal pytest coverage for:

- config validation behavior
- checkpoint save/resume behavior

## What a beginner should do here

- Run the tests after changing YAML/config validation, registry wiring, or trainer/checkpoint code.

## Key files

- `tests/conftest.py`: adds the repo root and `policy_constructor/` to `sys.path` for tests.
- `tests/test_config_validation.py`: validates error paths for common config mistakes.
- `tests/test_integration_resume.py`: runs a tiny fit + resumes from a saved checkpoint.

## How it connects to the pipeline

The tests call `offline_trainer.experiment.run_experiment()` directly (there is no CLI entrypoint).

## Common edits / extension points

- Add tests when you add new validation rules or new built-in components.

## Links

- [Root quickstart](../README.md#quickstart)
