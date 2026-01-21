# Example Configs

These YAML files are consumed by `trainer.api.train` and demonstrate common configuration patterns.

Configs:
- `minimal.yaml`: minimal smoke run with default components and a policy_constructor model config.
- `swap_optim_sched.yaml`: swaps the optimizer and scheduler to show registry usage.
- `custom_trainer_custom_data.yaml`: imports plugin modules and uses custom trainer and datamodule types.

Notes:
- The loader supports `defaults` composition and deep merge.
- Relative paths in `model.config_path` are resolved relative to the config file.
