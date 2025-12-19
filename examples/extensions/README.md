# Extension Examples

These modules show how to register custom components into the global registries.

Files:
- `custom_trainer.py`: registers a trainer via `TRAINER_REGISTRY`.
- `custom_data.py`: registers a datamodule via `DATAMODULE_REGISTRY`.

Use from YAML by adding the module path under `plugins:` so it is imported at startup.
