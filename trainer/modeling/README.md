# Modeling

Model construction is delegated to policy_constructor via a thin adapter.

Files:
- `factories.py`: `PolicyConstructorModelFactory` calls `model_constructor.build_model`.

Usage:
- Supply `model.config_path` (path to a policy_constructor YAML) or
  `model.config` (inline config dict) in your experiment YAML.
