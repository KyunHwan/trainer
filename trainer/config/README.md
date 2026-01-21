# Config

Configuration is YAML-driven and validated with pydantic.

Modules:
- `loader.py`: loads YAML with `defaults` composition and deep merge.
- `schemas.py`: pydantic models for `ExperimentConfig` and component specs.
- `errors.py`: structured validation errors with friendly formatting.

Notes:
- The loader resolves relative `defaults` paths against the current config file.
- Component specs use `{type, params}` to reference registry entries.
