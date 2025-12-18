# extensions/

## What this folder is

This is the recommended home for project-specific **plugins**.

Plugins are plain Python modules that are imported from YAML via the top-level `imports:` list.
If a module defines `register(registry)`, the model-constructor import mechanism will call it, allowing the module to add registry keys.

## What a beginner should do here

- Copy the structure of `example_plugin.py` when creating your own custom Trainer/DataModule/Loss/etc.
- Add your plugin module to YAML `imports:`.

## Key files

- `extensions/example_plugin.py`
  - registers `data.constant_regression`
  - registers `trainer.verbose`

## How it connects to the pipeline

- Your experiment YAML can contain:

  ```yaml
  imports:
    - extensions.example_plugin
  ```

- During model construction, the import hook runs and calls `register(registry)`.
- After that, `offline_trainer` can instantiate your custom components by referencing their `_type_` keys.

## Common edits / extension points

- Add new plugin modules under `extensions/` (e.g. `extensions/my_plugin.py`).
- Ensure `configs/base/runtime.yaml:settings.allowed_import_prefixes` includes `extensions.` (it does by default).
- Use globally unique registry keys (e.g. `trainer.my_project.custom_trainer`).

## Links

- [Root plugin overview](../README.md#registry--plugins-how-yaml-selects-components)
- [Example experiment using a plugin](../configs/experiments/custom_plugin.yaml)
