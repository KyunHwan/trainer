# trainer

Core package for config-driven training. The public entrypoint is `trainer.api.train`.

Key areas:
- `api.py`: loads config, registers components, instantiates modules, and launches training.
- `config/`: YAML loader, schema validation, and config error formatting.
- `registry/`: registry implementation, builtins, and plugin loader.
- `training/`: default training loop, checkpointing, AMP, and state tracking.
- `components/`: optimizer, scheduler, loss, metrics, callbacks, and loggers.
- `data/`: datamodule protocol and data utilities.
- `modeling/`: model factory adapter for policy_constructor.
- `utils/`: selection, device, tree utilities, seeding, and import helpers.
