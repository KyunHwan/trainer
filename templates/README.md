# Templates

These are minimal templates for custom components. Copy them into your own package (or `examples/extensions/`) and edit as needed.

Files:
- `callback_template.py`: callback skeleton with state hooks.
- `datamodule_template.py`: datamodule skeleton with dataloaders.
- `trainer_template.py`: trainer skeleton that can wrap your own loop.

Register components with the appropriate registry decorator and import the module via `plugins:` so the registry is populated at runtime.
