# Configuration reference

## Contents

- [The YAML loader](#the-yaml-loader)
- [defaults composition](#defaults-composition)
- [Pydantic schema](#pydantic-schema)
  - [ExperimentConfig](#experimentconfig-root)
  - [ModelConfig](#modelconfig)
  - [DataConfig](#dataconfig)
  - [TrainConfig](#trainconfig)
  - [ComponentSpec](#componentspec)
  - [OptimizerSpec and OptimizerParams](#optimizerspec-and-optimizerparams)
  - [ComponentConfigPaths](#componentconfigpaths)
  - [EMAConfig and CheckpointConfig](#emaconfig-and-checkpointconfig)
- [extra="allow" and undeclared fields](#extraallow-and-undeclared-fields)
- [Validation errors](#validation-errors)
- [Where each field is consumed](#where-each-field-is-consumed)

## The YAML loader

The loader is [`trainer/config/loader.py`](../trainer/config/loader.py). The public entry point is `load_config(path)`:

```python
def load_config(path: str) -> dict[str, Any]:
    abs_path = os.path.abspath(path)
    return _load_with_defaults(abs_path, stack=[])
```

It reads a YAML file with `yaml.safe_load`, recursively expands any `defaults:` list, and deep-merges. Relative paths in `defaults` entries are resolved against **the directory of the file declaring them**, not the working directory. Cycles are detected by tracking a stack of currently-loading paths and raising `ConfigLoadError("Config defaults cycle detected: ...")` on revisit.

## defaults composition

```yaml
defaults:
  - base: ./base.yaml         # processed first
  - augmentations: ./aug.yaml # then this, merged on top of base
# keys here override everything above
train:
  epoch: 200
```

Rules:

1. Each entry must be a **single-key mapping**; the key is a label (used only for readability), the value is the path.
2. The path can be relative or absolute. Relative resolves against the directory of the *current* YAML.
3. Entries are processed in list order. Each is loaded, deep-merged onto the running `merged` dict, and the next is merged on top.
4. After all defaults are merged, the current file's top-level keys are deep-merged on top — i.e. **the current file wins ties with its defaults**.

A worked example. Given `base.yaml`:

```yaml
train:
  epoch: 100
  save_every: 5
```

and `child.yaml`:

```yaml
defaults:
  - base: ./base.yaml
train:
  epoch: 200
```

Loading `child.yaml` returns `{"train": {"epoch": 200, "save_every": 5}}` — `train.epoch` is overridden by the child, `train.save_every` is inherited from the base.

`_deep_merge` recurses into nested dicts; for non-dict leaves (lists, scalars), the override replaces the base value outright. **Lists are not concatenated** — you cannot extend `plugins:` via defaults, you must override the whole list.

## Pydantic schema

After loading, `validate_config(raw)` validates the dict against `ExperimentConfig`. Schema is in [`trainer/config/schemas.py`](../trainer/config/schemas.py). All models use `model_config = ConfigDict(extra="allow")` unless noted (see [§extra="allow"](#extraallow-and-undeclared-fields)).

### ExperimentConfig (root)

| Field | Type | Default | Notes |
|---|---|---|---|
| `plugins` | `list[str]` | *(required)* | Module paths passed to `load_plugins` |
| `seed` | `int \| None` | `123` | Base RNG seed (`train.seed` is what the trainer actually reads — this top-level field is *declared* but the trainer reads `getattr(config.train, "seed", 0)`) |
| `deterministic` | `bool` | `False` | Declared but not consumed by either entrypoint |
| `device` | `str` | `"auto"` | Declared but not consumed; device selection is rank-based |
| `model` | `ModelConfig` | *(required)* | See below |
| `data` | `DataConfig` | *(required)* | See below |
| `train` | `TrainConfig` | *(required)* | See below |

### ModelConfig

| Field | Type | Default | Notes |
|---|---|---|---|
| `find_unused_parameters` | `bool` | *(required, no default)* | Passed to `DDP(..., find_unused_parameters=...)` |
| `component_config_paths` | `ComponentConfigPaths` | *(required)* | Mapping `{name: yaml_path}`. See below |
| `component_build_args` | *(allowed extra)* | — | `dict[name, {init: bool, freeze: bool, online_update?: bool}]`. Read directly by `_build_models` |
| `component_optims` | *(allowed extra)* | — | `dict[name, {type: str, params: dict}]`. Read by `_build_optimizers` |

`component_build_args` and `component_optims` are **not** declared in the Pydantic model; they pass through because of `extra="allow"`. Validation will not catch typos or missing fields in those dicts.

### DataConfig

| Field | Type | Default | Notes |
|---|---|---|---|
| `datamodule` | `ComponentSpec` | *(required)* | `{type, params}`. `type` keys into `DATASET_BUILDER_REGISTRY` |
| `batch_size` | — | — | Used by `_build_dataloader`; not declared (extra=allow) |
| `num_workers` | — | — | Same |
| `pin_memory` | — | — | Same |
| `persistent_workers` | — | — | Same |
| `prefetch_factor` | — | — | Same |

The `data:` block currently relies on `extra="allow"` for everything except the `datamodule` subblock.

### TrainConfig

`TrainConfig` declares many fields the entrypoints don't read, plus reads several it doesn't declare. Declared, validated fields:

| Field | Type | Default | Validator |
|---|---|---|---|
| `trainer` | `ComponentSpec` | *(required)* | `type` keys into `TRAINER_REGISTRY` |
| `optimizer` | `OptimizerSpec \| None` | `None` | Declared but unused (optimizers are per-component, in `model.component_optims`) |
| `scheduler` | `ComponentSpec` | `ComponentSpec(type="none")` | Declared but unused |
| `loss` | `ComponentSpec \| None` | `None` | `type` keys into `LOSS_BUILDER_REGISTRY` |
| `metrics` | `list[ComponentSpec]` | `[]` | Declared but unused |
| `callbacks` | `list[ComponentSpec]` | `[]` | Declared but unused |
| `loggers` | `list[ComponentSpec]` | `[ComponentSpec(type="noop")]` | Declared but unused |
| `model_input` | `str \| int \| None` | `None` | Declared but unused |
| `max_epochs` | `int` | `1` | `> 0`. Declared but unused (`train.epoch` is what's read) |
| `max_steps` | `int \| None` | `None` | `> 0` if set. Declared but unused |
| `accumulate_grad_batches` | `int` | `1` | `> 0`. Declared but unused |
| `amp` | `bool` | `False` | Declared but unused (autocast is always on) |
| `gradient_clip_val` | `float \| None` | `None` | `>= 0` if set. Declared but unused (trainers clip themselves) |
| `log_every_n_steps` | `int` | `1` | `> 0`. Declared but unused (logging is every iteration) |
| `ema` | `EMAConfig` | `EMAConfig()` | Declared but unused |
| `checkpoint` | `CheckpointConfig` | `CheckpointConfig()` | Declared but unused (trainers use the top-level `save_dir`, `save_every`, `load_dir`) |

Fields the entrypoints actually read from `train:` — all going through `extra="allow"`, so no validation:

| Read via | Used for |
|---|---|
| `config.train.project_name` | wandb run name |
| `config.train.save_dir` | Checkpoint and stats output directory |
| `config.train.save_every` | Epoch interval for saves (offline) / iteration multiplier (online) |
| `config.train.epoch` | Number of epochs (offline) |
| `getattr(config.train, "seed", 0)` | Base seed (overrides top-level `seed`) |
| `config.train.load_dir` | Resume directory; may be `null` |

> TODO (maintainer): the schema has drifted from what the trainers actually read. Either move the active fields into validated `TrainConfig` declarations or strip the unused ones. Documenting the gap as-is until that cleanup happens.

### ComponentSpec

```python
class ComponentSpec(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str                                  # validator: non-empty
    params: dict[str, Any] = {}
```

The `{type, params}` shape is reused for trainers, losses, optimizers, schedulers, and the datamodule. The `type` is the registry key; `params` is forwarded to the constructor via `_filter_kwargs` in [`utils/import_utils.py`](../trainer/utils/import_utils.py).

### OptimizerSpec and OptimizerParams

`OptimizerSpec` is `ComponentSpec` with `params: OptimizerParams`.

```python
class OptimizerParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    lr: float | None = None                    # validator: > 0 if set
```

`extra="allow"` is what lets each optimizer accept its own keys (`peak_lr`, `betas`, `weight_decay`, `total_steps`, `warmup_steps`, `start_lr`, `end_lr`, etc.). The `lr` validator is mostly defensive — most optimizers in the codebase use `peak_lr` or `max_lr` instead, but `lr` is still validated if present.

`OptimizerParams` is used in two places: as the type of `OptimizerSpec.params`, and explicitly via `OptimizerParams.model_validate(config.model.component_optims[name]['params'])` in `_build_optimizers`. The second invocation is what actually runs at runtime — the schema-time validation through `OptimizerSpec` doesn't fire because `component_optims` itself lives under `extra="allow"`.

### ComponentConfigPaths

```python
class ComponentConfigPaths(RootModel[dict[str, str]]):
    @model_validator(mode="after")
    def _validate_entries(self):
        # must be non-empty, all keys non-empty strings, all values non-empty strings
```

It's a `RootModel` wrapping `dict[str, str]`. `.as_dict()` returns a plain dict for downstream consumption.

### EMAConfig and CheckpointConfig

Both declared (with `extra="forbid"`), both unused by current entrypoints.

```python
class EMAConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    decay: float = 0.999                       # validator: in (0, 1)

class CheckpointConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    save_dir: str = "runs"
    save_every_n_steps: int | None = None      # validator: > 0 if set
    save_last: bool = True
    resume_from: str | None = None
```

Trainers currently do not use these. Future EMA / step-based checkpointing would live here.

## extra="allow" and undeclared fields

`extra="allow"` lets you put arbitrary keys in a YAML that aren't declared on the Pydantic model. They survive validation and become attributes on the model instance, accessible via `getattr(...)`. The downside is you get **no validation** for them — typos go through silently.

In practice the codebase relies on `extra="allow"` for `train.project_name`, `train.save_dir`, `train.save_every`, `train.epoch`, `train.seed`, `train.load_dir`, all of `data.*` other than `datamodule`, and the whole `component_build_args` / `component_optims` substructure. If you mistype `save_evry`, no error is raised — the trainer just never finds the key and falls back to whatever default `getattr` returns (often `0` or `None`), which can manifest as no saves happening at all.

When you add fields here, name them precisely and double-check spelling against `offline_trainer.py`.

## Validation errors

When `validate_config` fails, it raises `ConfigError` (in [`trainer/config/errors.py`](../trainer/config/errors.py)):

```text
Config validation failed with 2 error(s):
- error_path: model.find_unused_parameters
  error_message: Field required
- error_path: train.trainer.type
  error_message: type must be a non-empty string
```

The format is one entry per Pydantic error, with the dotted YAML path and the validator's message. `ConfigValidationIssue` is the underlying dataclass (`error_path`, `error_message`, optional `hint`). The list is also exposed on the exception as `exc.issues` if you want to handle errors programmatically.

`ConfigLoadError` (raised by the loader, before validation) reports file-loading problems: missing files, non-mapping root, malformed `defaults` entries, cycle detection.

## Where each field is consumed

A reference of "which helper reads this field". All paths are in [`trainer/offline_trainer.py`](../trainer/offline_trainer.py) (the online trainer reads the same fields from the same helpers).

| Field | Reader |
|---|---|
| `plugins` | `train()` calls `load_plugins(config.plugins)` |
| `model.find_unused_parameters` | `_build_models` via `getattr(config.model, "find_unused_parameters", False)` |
| `model.component_config_paths` | `_build_models` resolves to absolute paths and passes to `PolicyConstructorModelFactory.build` |
| `model.component_build_args[name].init` / `.freeze` | `_build_models` |
| `model.component_build_args[name].online_update` | online trainer's save block only |
| `model.component_optims[name].type` / `.params` | `_build_optimizers` |
| `data.datamodule.type` / `.params` | `_build_dataloader` (registry lookup + factory instantiation) |
| `data.batch_size`, `data.num_workers`, `data.pin_memory`, `data.persistent_workers`, `data.prefetch_factor` | `_build_dataloader` |
| `train.trainer.type` / `.params` | `_build_trainer` |
| `train.loss.type` / `.params` | `_build_loss` |
| `train.project_name` | `wandb.init(name=...)` (offline only) |
| `data.datamodule.params.task_name` | `wandb.init(project=...)` |
| `train.save_dir` | `_build_dataloader`, `_save_checkpoints`, `train()` |
| `train.save_every` | `train()` epoch check (offline) / iteration check × 25 (online) |
| `train.epoch` | `train()` outer loop count (offline only) |
| `train.seed` | Both phases of seed protocol |
| `train.load_dir` | `_build_models` and `_build_optimizers` resume logic |
