# Testing

## State of the suite

There is currently **no automated test suite** in this repository. The previous top-level README referenced `tests/` and `pytest` workflows, but no `tests/` directory or `pytest.ini` exists on disk at the time of this writing. The `pytest` package is installed via [`env_setup.sh`](../env_setup.sh), but only because other dependencies pull it in transitively.

Running `pytest` from the repo root currently does nothing — there are no test files to collect.

> TODO (maintainer): decide whether to (a) restore a `tests/` directory with smoke tests for the config loader, registry, plugin loader, and a minimal `train()` invocation, or (b) update the top-level README to reflect that automated tests are out of scope for this repo. This doc will be updated to match whichever path is taken.

## Verifying changes manually

In the absence of automated tests, this is the manual checklist used to verify a change end-to-end:

1. **Config load** — run `python -c "from trainer.trainer.config.loader import load_config; from trainer.trainer.config.schemas import validate_config; print(validate_config(load_config('<path/to/config.yaml>')))"`. Confirms YAML composition and Pydantic validation produce a sane `ExperimentConfig`.
2. **Plugin import** — for every entry in `plugins:`, run `python -c "import <module>"`. Confirms each plugin file imports without error.
3. **Registry presence** — after importing the plugins, confirm `TRAINER_REGISTRY.has("<key>")`, etc., for every type key in the YAML.
4. **First-batch smoke** — launch with a tiny `train.epoch: 1` and small `data.batch_size` (e.g. 2) and confirm one forward / backward / step completes and a wandb metric is logged. This is the minimum "the framework still wires up" check.
5. **Checkpoint round-trip** — run with `train.save_every: 1` for one epoch, then restart with `train.load_dir` pointing at the produced `epoch_1/`. Confirm no `doesn't exist as a file!` warnings.

These checks are not a substitute for unit tests, but they catch ~90% of regressions in practice.

## If you add tests

A reasonable starting test surface, in order of value:

| Test target | Why it matters |
|---|---|
| `load_config` with `defaults:` composition and cycle detection | The composition logic is non-trivial and easy to regress |
| `validate_config` rejecting malformed YAMLs | The Pydantic schema is currently the main "is this YAML sensible" check |
| `Registry.register` rejecting duplicate keys | Silent registration overwrites would be very confusing |
| `load_plugins` deduplication of already-loaded modules | Re-importing would re-register and raise |
| `_build_models` with a fake `PolicyConstructorModelFactory` | Tests the init / freeze / SyncBN / DDP-wrap branches without needing real model YAMLs |
| `_save_checkpoints` + `_build_models` resume round-trip | Catches checkpoint-format regressions |

If you add a `tests/` directory, put a `conftest.py` at its root and use `pytest --rootdir=.` from the repo root. Reference any new tests from this doc once they exist.
