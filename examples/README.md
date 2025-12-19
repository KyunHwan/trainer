# Examples

This directory contains runnable examples that exercise the core training flow and the plugin system.

Contents:
- `configs/`: YAML configs used by `offline_trainer.api.train`.
- `extensions/`: importable modules that register custom components.
- `run_from_python.py`: minimal entrypoint without a CLI wrapper.

Try the minimal run:
```
python examples/run_from_python.py
```
