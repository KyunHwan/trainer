# Core concepts

*Estimated reading time: 15 minutes*

This document is the conceptual spine of the framework. If you read only one doc beyond the getting-started guide, read this one. Every other reference doc assumes the vocabulary defined here.

## Contents

- [The shape of the framework](#the-shape-of-the-framework)
- [Registries](#registries)
- [Plugins](#plugins)
- [Protocols (templates)](#protocols-templates)
- [Factories](#factories)
- [The nn.ModuleDict convention](#the-nnmoduledict-convention)
- [Per-component optimizers, init, freeze](#per-component-optimizers-init-freeze)
- [The data dict](#the-data-dict)
- [The stats dict and where normalization lives](#the-stats-dict-and-where-normalization-lives)
- [The train_step contract](#the-train_step-contract)

## The shape of the framework

The training entrypoint does exactly one thing, in this order:

```text
YAML config
  → load_config (compose defaults, deep-merge)
  → validate_config (Pydantic → ExperimentConfig)
  → load_plugins (importlib.import_module per plugin path → @register decorators fire)
  → _build_trainer
       ├─ _build_models (PolicyConstructorModelFactory → nn.ModuleDict)
       ├─ _build_optimizers (one optimizer per non-frozen component)
       └─ _build_loss
  → _build_dataloader (DatasetFactory.build → norm_stats persisted to disk)
  → epoch loop:
      for batch:
        cast_dtype → move_to_device → autocast(bfloat16)
          → trainer.train_step(data, stats)
        if rank == 0: wandb.log(loss_dict)
```

The reason this layout is worth understanding before you read any specific file: every named class in this codebase is a **plug-in point** for that pipeline. There are four families of plug-ins (trainer, dataset factory, loss factory, optimizer factory), four registries holding them, four protocols defining their contracts, and a single YAML field (`plugins:`) that lists which modules to import so their `@register` decorators run.

## Registries

A **registry** is a typed `dict[str, type]` with a decorator for registration and a `get(key)` method for lookup. The implementation is [`trainer/registry/core.py`](../trainer/registry/core.py):

```python
class Registry(Generic[T]):
    def register(self, key=None) -> Callable[[type[T]], type[T]]:
        def decorator(cls):
            self.add(key or cls.__name__, cls)
            return cls
        return decorator
```

`Registry.add` enforces an optional `expected_base` constraint — when set, `issubclass(cls, expected_base)` must hold or registration raises `TypeError`. For Protocol bases (see below), this is a structural check.

[`trainer/registry/__init__.py`](../trainer/registry/__init__.py) instantiates the four global registries used throughout the codebase:

| Registry | Holds | Looked up by |
|---|---|---|
| `TRAINER_REGISTRY` | Subclasses of the `Trainer` protocol | `train.trainer.type` |
| `DATASET_BUILDER_REGISTRY` | Subclasses of `DatasetFactory` | `data.datamodule.type` |
| `OPTIMIZER_BUILDER_REGISTRY` | Subclasses of `OptimizerFactory` | Each entry in `model.component_optims[name].type` |
| `LOSS_BUILDER_REGISTRY` | Subclasses of `LossFactory` | `train.loss.type` |

The training loop calls `TRAINER_REGISTRY.get(config.train.trainer.type)` (and analogous lookups for the others). A missing key raises `KeyError` with a list of registered keys — the most common cause is forgetting to list the component's module in `plugins:`.

## Plugins

Registries are global but empty until something imports the modules that contain the `@register` decorators. [`trainer/registry/plugins.py`](../trainer/registry/plugins.py) is the loader:

```python
def load_plugins(modules: Iterable[str]) -> None:
    for module in modules:
        if module in _LOADED_MODULES:
            continue
        importlib.import_module(module)
        _LOADED_MODULES.add(module)
```

It is called once, near the top of `train()`, with the YAML's `plugins:` list:

```yaml
plugins:
  - "experiment_training.components.dataloader.lerobot_data"
  - "experiment_training.components.loss.sinkhorn_knopp"
  - "experiment_training.components.optimizer.adamw_cosine_decay"
  - "experiment_training.components.trainer.imitation_learning.vfp_single_expert.vfp_single_expert_trainer"
```

Each string is a Python import path. Importing the module triggers any module-scope `@TRAINER_REGISTRY.register(...)`, `@LOSS_BUILDER_REGISTRY.register(...)`, etc. Until that import happens, the corresponding registry key does not exist. **The most common cause of `KeyError: <X> registry has no key '<Y>'` is that the module declaring `Y` is not listed in `plugins:`.**

`_LOADED_MODULES` is a process-local set that prevents re-import. Re-importing the same module would also re-run its `@register` decorators, which would then raise `KeyError: registry already has key '<Y>'` from `Registry.add`.

## Protocols (templates)

[`trainer/templates/`](../trainer/templates/) declares four `@runtime_checkable` Protocols (PEP 544). They are *structural* interfaces — any class with the matching methods qualifies, without inheriting from the protocol.

| Protocol | File | Methods | What the framework expects |
|---|---|---|---|
| `Trainer` | [`templates/trainer.py`](../trainer/templates/trainer.py) | `__init__(models, optimizers, loss)`, `train_step(data, stats) -> dict[str, Any]` | Constructed once per run; called once per batch |
| `DatasetFactory` | [`templates/dataset.py`](../trainer/templates/dataset.py) | `build(**kwargs) -> dict[str, Any]` | Called once at startup; returns `{"dataset": Dataset, "norm_stats": dict}` |
| `LossFactory` | [`templates/loss.py`](../trainer/templates/loss.py) | `build() -> nn.Module` | Called once at startup; result is moved to the training device |
| `OptimizerFactory` | [`templates/optim.py`](../trainer/templates/optim.py) | `build(params: Iterable[nn.Parameter]) -> torch.optim.Optimizer` | Called once per non-frozen model component |

Two pieces of fine print:

- **Trainer.__init__ also receives `device`** in practice. The protocol doesn't list it, but every concrete trainer accepts it. `instantiate` in [`trainer/utils/import_utils.py`](../trainer/utils/import_utils.py) filters constructor kwargs by signature, so `device=...` is forwarded only when accepted. Documenting on a case-by-case basis is fine; structural conformance is what `isinstance(trainer, Trainer)` checks at startup.
- **The check is `isinstance(trainer, Trainer)`** in `_build_trainer`. Because `Trainer` is `@runtime_checkable`, this returns `True` for any object with `train_step` and `__init__` callable — signatures are not introspected. If your trainer's `train_step` has the wrong parameters, you'll find out at call time (likely a `TypeError: missing required positional argument`).

## Factories

A *factory* is the class registered in a registry; the object it produces is the thing that actually runs. The split exists because the registered class needs to be configured from YAML *before* it can construct the runtime object.

The canonical example is [`adamw_cosine_decay.py`](../experiment_training/components/optimizer/adamw_cosine_decay.py):

```python
@OPTIMIZER_BUILDER_REGISTRY.register("adamw_warmup_cosine_decay")
class AdamW_WarmupCosine_Builder(nn.Module):
    def __init__(self, peak_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                 *, total_steps, warmup_steps, start_lr, end_lr):
        ...  # store the hyperparameters

    def build(self, params) -> AdamWWithWarmupCosine:
        return AdamWWithWarmupCosine(params=params, peak_lr=self.peak_lr, ...)
```

The training loop instantiates the factory from `component_optims[name].params`, then calls `factory.build(model.parameters())`. Splitting like this is what lets the YAML say *what the optimizer should look like* without having access to the model's parameters yet.

The same shape applies to losses (`LossFactory.build()`), datasets (`DatasetFactory.build(opt_params, params)`), and trainers — though for trainers, the framework constructs the registered class directly (the trainer *is* the runtime object; there's no separate `build` step).

[`PolicyConstructorModelFactory`](../trainer/modeling/factories.py) is a slightly different factory: it is not registered (there's only one model construction backend), but it follows the same pattern — `factory.build(model_cfg_dict)` returns either a single `nn.Module` or a `dict[name -> nn.Module]` depending on how the input dict is shaped. For experiment configs, the input is a dict of named component paths, and the output is a dict of named `GraphModel` instances.

## The nn.ModuleDict convention

After `_build_models` finishes, you don't have *a model* — you have an `nn.ModuleDict`:

```python
models = nn.ModuleDict({
    "head_backbone":  GraphModel(...),
    "left_backbone":  GraphModel(...),
    "right_backbone": GraphModel(...),
    "info_embedder":  GraphModel(...),
    "action_decoder": GraphModel(...),
})
```

Three reasons it's built this way rather than as a single composed `nn.Module`:

1. **Each component has its own lifecycle**. `init`, `freeze`, and "give me my own optimizer" are decided per component via `model.component_build_args[name]` and `model.component_optims[name]`. With one monolithic model, you'd need to manually identify which submodule to freeze, which to init, which to assign learning rates to.
2. **DDP wraps each entry independently**. Frozen components are *not* wrapped (they have no gradients to all-reduce). Non-frozen components are each wrapped in their own `DDP(...)`. This is what allows you to have a frozen 1B-parameter Depth Anything v3 model alongside a 20M-parameter action decoder you're actively training.
3. **The trainer can choose how components interact**. `train_step` reads `self.models["head_backbone"]`, `self.models["action_decoder"]`, etc., and decides what to compose with what. The framework doesn't know or care about the topology.

Frozen components live in the same `ModuleDict` but `policy.eval()` + `requires_grad_(False)` is applied, so they don't accumulate gradients and their BN/Dropout layers behave correctly during inference inside `train_step`.

## Per-component optimizers, init, freeze

Three dictionaries on `model:` drive component lifecycle:

```yaml
model:
  find_unused_parameters: true
  component_config_paths:
    head_backbone:  "experiment_models/vfp_single_expert/exp1/head_backbone.yaml"
    info_embedder:  "experiment_models/vfp_single_expert/exp1/info_embedder.yaml"
    action_decoder: "experiment_models/vfp_single_expert/exp1/action_decoder.yaml"
  component_build_args:
    head_backbone:  { init: false, freeze: false }   # use the backbone's loaded weights, train it
    info_embedder:  { init: true,  freeze: false }   # Xavier/Kaiming init, train it
    action_decoder: { init: true,  freeze: false }
  component_optims:
    head_backbone:
      type: "adamw_warmup_cosine_decay"
      params: { peak_lr: 1.0e-4, ... }
    info_embedder:
      type: "adamw_warmup_cosine_decay"
      params: { peak_lr: 1.0e-4, ... }
    action_decoder:
      type: "adamw_warmup_cosine_decay"
      params: { peak_lr: 1.0e-4, ... }
```

The keys in all three dictionaries must match. The behavior at startup, per component (in `_build_models`):

- If `train.load_dir` is set and `{load_dir}/{name}.pt` exists → load weights, skip `init`.
- Else if `init: true` → apply `init_weights` (Kaiming for Linear/ReLU, Xavier for Conv2d).
- If `freeze: true` → set `requires_grad_(False)`, `.eval()`, move to device, **skip DDP wrap and skip optimizer creation**. Done.
- Else → optionally convert BN to SyncBN, move to device, wrap in DDP, create an optimizer from `component_optims[name]`.

Optimizers for frozen components are silently skipped — even if you list them in `component_optims`, no optimizer is built because `[p for p in models[name].parameters() if p.requires_grad]` is empty.

## The data dict

The dataset factory returns a `Dataset` whose `__getitem__` yields a dict. After the dataloader batches and the trainer's per-step pipeline runs (`cast_dtype → move_to_device → autocast`), the trainer's `train_step` receives a dict with roughly these keys (the canonical LeRobot-shaped batch):

| Key | Shape (typical) | Meaning |
|---|---|---|
| `action` | `(B, action_horizon, action_dim)` | Future action chunk (target) |
| `observation.state` | `(B, T_obs, state_dim)` | Proprioceptive state history |
| `observation.current` | `(B, T_obs, state_dim)` | Most-recent state (used for relative observation framing) |
| `observation.images.cam_head` | `(B, T_img, C, H, W)` | Head camera RGB frames |
| `observation.images.cam_left`, `cam_right` | same | Left / right cameras |
| `labels.reward` | `(B, reward_horizon)` | Per-step reward (RL only) |
| `task_index` | `(B,)` | Integer index used to look up natural-language prompts |

`task_index` is used by some trainers as an index into a `tasks.parquet` file shipped with the LeRobot dataset — each row is a `(task_index, prompt_string)` pair. Trainers that need text conditioning (e.g. OpenPI-batched) read the dataset's `tasks` table to translate the index back to a prompt before passing it to a tokenizer.

> TODO (maintainer): confirm the schema of `tasks.parquet` (column names, dtypes) — the LeRobot version-pinning has been moving. The exact lookup mechanism is buried in `LeRobotDataset` and not reproduced in our code.

## The stats dict and where normalization lives

The dataset factory may return a second key, `"norm_stats"`:

```python
return {
    "dataset": dataset,
    "norm_stats": dataset.meta.stats,   # for LeRobot, a dict keyed by data column
}
```

The shape is `{key: {"mean": <list-or-tensor>, "std": <list-or-tensor>, "min": ..., "max": ...}}`. Rank 0 pickles it to `{save_dir}/dataset_stats.pkl`; all ranks then convert it to GPU tensors via `tree_map(map_list_to_torch, stats) → cast_dtype(..., torch.float32) → move_to_device(..., device)` once before the loop. The result is `stats_gpu`, passed into `trainer.train_step(data=data, stats=stats_gpu)` every step.

**Normalization itself is the trainer's responsibility, not the loop's.** The current `train()` function in [`offline_trainer.py`](../trainer/offline_trainer.py) does **not** subtract means or divide by standard deviations — it only builds `stats_gpu` and hands it to `train_step`. Whether a given trainer normalizes inputs, outputs, both, or neither is a trainer-implementation choice. Trainers that follow the LeRobot convention will typically apply:

```python
data["action"] = (data["action"] - stats["action"]["mean"]) / (stats["action"]["std"] + 1e-8)
data["observation.state"] = (data["observation.state"] - stats["observation.state"]["mean"]) \
                            / (stats["observation.state"]["std"] + 1e-8)
```

near the top of their forward, but this is convention rather than enforcement. When you write a new trainer, decide explicitly: normalize inside `train_step`, or document that your trainer expects pre-normalized inputs.

## The train_step contract

Per [`trainer/offline_trainer.py`](../trainer/offline_trainer.py) and [`trainer/online_trainer.py`](../trainer/online_trainer.py), the call site is:

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    loss_dict = trainer.train_step(data=data, stats=stats_gpu)
```

That fixes the contract:

- **Signature**: `train_step(self, data: dict[str, Any], stats: Any) -> dict[str, Any]`
- **Return**: a dict of metric values. Tensors are detached (`.detach().item()`) before logging; non-tensor scalars are logged as-is. `'epoch'` is appended automatically (= `iterations / num_iter_per_epoch`).
- **Side effects expected inside**: `optimizer.zero_grad()`, forward, `loss.backward()`, optional `torch.nn.utils.clip_grad_norm_`, `optimizer.step()`. The training loop does **not** call any of these on your behalf.
- **Mutation rules**: `data` and `stats` are not used by the framework after `train_step` returns, so in-place modification is safe. The framework does not retain references.

A worked example is in [07_extending.md](07_extending.md). Existing trainer implementations in [`experiment_training/components/trainer/`](../experiment_training/components/trainer/) follow this pattern; some declare additional positional parameters (`epoch`, `total_epochs`, `iterations`) that are **not** supplied by the current loop, so they're effectively dead arguments. If you're authoring a new trainer, match the canonical signature.
