# Extending the framework

How to add a new dataset, trainer, loss, or optimizer. Each recipe shows the file to create, the class skeleton, the registration call, the YAML hook, and the framework's runtime expectations.

## Contents

- [The shape of every recipe](#the-shape-of-every-recipe)
- [Recipe: a new dataset factory](#recipe-a-new-dataset-factory)
- [Recipe: a new trainer](#recipe-a-new-trainer)
- [Recipe: a new loss](#recipe-a-new-loss)
- [Recipe: a new optimizer](#recipe-a-new-optimizer)
- [Wiring it up: the plugins list](#wiring-it-up-the-plugins-list)
- [Worked example end-to-end](#worked-example-end-to-end)

## The shape of every recipe

Each new component is one Python file containing:

1. A `@<REGISTRY>.register("<key>")` decorator on a class.
2. The methods required by the corresponding protocol in [`trainer/templates/`](../trainer/templates/).

You then add the file's module path to the `plugins:` list in your training YAML, and reference the registered key by string in the appropriate config field.

The framework discovers the new component because importing the module runs the decorator. No imports, no registration.

## Recipe: a new dataset factory

**File**: `experiment_training/components/dataloader/my_dataset.py` (or anywhere importable).

**Protocol**: `DatasetFactory` in [`trainer/templates/dataset.py`](../trainer/templates/dataset.py).

```python
from typing import Any
from torch.utils.data import Dataset
from trainer.trainer.registry import DATASET_BUILDER_REGISTRY


@DATASET_BUILDER_REGISTRY.register("my_dataset")
class MyDatasetFactory:
    def build(self, opt_params: dict[str, Any] | None, params: dict[str, Any]) -> dict[str, Any]:
        # opt_params (provided by the trainer): {'local_rank': int, 'dist_enabled': bool, 'save_dir': str}
        # params (from YAML data.datamodule.params): your factory's hyperparameters

        dataset: Dataset = build_my_torch_dataset(params)

        # If you have normalization statistics, return them under "norm_stats".
        # Shape: {key: {"mean": list-or-tensor, "std": list-or-tensor}, ...}
        norm_stats = {
            "action":            {"mean": [...], "std": [...]},
            "observation.state": {"mean": [...], "std": [...]},
        }
        return {"dataset": dataset, "norm_stats": norm_stats}
```

**At runtime** ([`_build_dataloader`](../trainer/offline_trainer.py)):

- `MyDatasetFactory` is instantiated. The factory's constructor receives the YAML `params` (kwargs are filtered to declared parameters; pass anything via `params` if your factory needs extra args).
- `factory.build(opt_params, params)` is called. `opt_params['local_rank']`, `['dist_enabled']`, `['save_dir']` are filled in by the trainer.
- The returned `"dataset"` is passed into `DataLoader`. The returned `"norm_stats"` (if present) is pickled to `{save_dir}/dataset_stats.pkl` by rank 0 and converted to GPU tensors at the start of the loop.

**Skipping norm_stats**: return only `{"dataset": dataset}` (or return the dataset directly without wrapping in a dict — the loop handles both). Then your trainer should not assume `stats` contains anything meaningful.

**YAML hook**:

```yaml
plugins:
  - "experiment_training.components.dataloader.my_dataset"

data:
  datamodule:
    type: "my_dataset"
    params:
      ...your factory's params...
```

## Recipe: a new trainer

**File**: `experiment_training/components/trainer/imitation_learning/my_trainer/my_trainer.py`.

**Protocol**: `Trainer` in [`trainer/templates/trainer.py`](../trainer/templates/trainer.py). The canonical signature is **what the entrypoint actually calls**, not what the protocol literally declares — see [04_concepts.md § The train_step contract](04_concepts.md#the-train_step-contract).

```python
import torch
import torch.nn as nn
from typing import Any
from trainer.trainer.registry import TRAINER_REGISTRY


@TRAINER_REGISTRY.register("my_trainer")
class MyTrainer(nn.Module):
    def __init__(self, *, models: nn.ModuleDict, optimizers: dict[str, torch.optim.Optimizer],
                 loss: nn.Module, device):
        super().__init__()
        self.models = models
        self.optimizers = optimizers
        self.loss = loss
        self.device = device

    def train_step(self, data: dict[str, Any], stats: Any) -> dict[str, Any]:
        # 1. Optional: normalize inputs using stats. The framework does NOT normalize.
        # data['action'] = (data['action'] - stats['action']['mean']) / (stats['action']['std'] + 1e-8)

        # 2. Zero grads. Required - the loop does not call this.
        for opt in self.optimizers.values():
            opt.zero_grad(set_to_none=True)

        # 3. Forward.
        features = self.models['backbone'](data['observation.images.cam_head'])
        pred     = self.models['decoder'](features, data['observation.state'])
        loss_value = self.loss(pred, data['action'])

        # 4. Backward.
        loss_value.backward()

        # 5. Optional: gradient clipping.
        for name in self.optimizers.keys():
            torch.nn.utils.clip_grad_norm_(self.models[name].parameters(), max_norm=1.0)

        # 6. Step.
        for opt in self.optimizers.values():
            opt.step()

        # 7. Return a flat dict of scalars to log. Tensors are auto-detached by _record.
        return {"loss": loss_value, "lr": self.optimizers['decoder'].param_groups[0]['lr']}
```

**At runtime** ([`_build_trainer`](../trainer/offline_trainer.py)):

- The class is instantiated with `models`, `optimizers`, `loss`, `device` as kwargs (plus any `train.trainer.params` from YAML, filtered by signature).
- The framework verifies `isinstance(trainer, Trainer)` after construction. The check is structural; it's satisfied as long as your class has `__init__` and `train_step` methods.
- The loop calls `trainer.train_step(data=data, stats=stats_gpu)` once per batch, under `torch.autocast(dtype=torch.bfloat16)`.

**Return value rules**: keys become wandb metric names. Tensor values must be scalar (the logger calls `.detach().item()`); a non-scalar tensor will raise. Python scalars (`float`, `int`, `bool`) pass through unchanged.

**YAML hook**:

```yaml
plugins:
  - "experiment_training.components.trainer.imitation_learning.my_trainer.my_trainer"

train:
  trainer:
    type: "my_trainer"
    params: {}    # additional kwargs to MyTrainer.__init__, if any
```

## Recipe: a new loss

**File**: `experiment_training/components/loss/my_loss.py`.

**Protocol**: `LossFactory` in [`trainer/templates/loss.py`](../trainer/templates/loss.py).

```python
import torch
import torch.nn as nn
from trainer.trainer.registry import LOSS_BUILDER_REGISTRY


@LOSS_BUILDER_REGISTRY.register("my_loss")
class MyLossFactory:
    def __init__(self, weight: float = 1.0, reduction: str = "mean"):
        self.weight = weight
        self.reduction = reduction

    def build(self) -> nn.Module:
        return MyLossModule(weight=self.weight, reduction=self.reduction)


class MyLossModule(nn.Module):
    def __init__(self, weight: float, reduction: str):
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = (pred - target).pow(2)
        return self.weight * (diff.mean() if self.reduction == "mean" else diff.sum())
```

**At runtime** ([`_build_loss`](../trainer/offline_trainer.py)):

- The factory is instantiated with the YAML's `train.loss.params`.
- `factory.build()` is called. If the result is an `nn.Module`, the loop moves it to `device`. The result becomes the `loss` argument to your trainer's `__init__`.
- The trainer is responsible for actually calling the loss. There is no framework-level "loss step".

**YAML hook**:

```yaml
plugins:
  - "experiment_training.components.loss.my_loss"

train:
  loss:
    type: "my_loss"
    params:
      weight: 1.0
      reduction: "mean"
```

## Recipe: a new optimizer

**File**: `experiment_training/components/optimizer/my_optimizer.py`.

**Protocol**: `OptimizerFactory` in [`trainer/templates/optim.py`](../trainer/templates/optim.py).

```python
from typing import Iterable
import torch
import torch.nn as nn
from torch.optim import AdamW
from trainer.trainer.registry import OPTIMIZER_BUILDER_REGISTRY


@OPTIMIZER_BUILDER_REGISTRY.register("my_optimizer")
class MyOptimizerFactory(nn.Module):
    def __init__(self, lr: float, weight_decay: float = 0.0):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay

    def build(self, params: Iterable[nn.Parameter]) -> torch.optim.Optimizer:
        return AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
```

**At runtime** ([`_build_optimizers`](../trainer/offline_trainer.py)):

- One optimizer is built per entry in `model.component_optims`. The factory is instantiated from `component_optims[name].params`, then `build(model.parameters())` is called.
- Frozen components are skipped (their `models[name].parameters()` has no `requires_grad=True` entries).
- If `train.load_dir` is set and `<load_dir>/<name>_opt.pt` exists, it's loaded via `optimizers[name].load_state_dict(...)`.

**Integrated schedulers**: if your optimizer owns its scheduler (like [`adamw_warmup_cosine_decay`](../experiment_training/components/optimizer/adamw_cosine_decay.py)), step the scheduler inside `optimizer.step()` and include scheduler state in `state_dict()`/`load_state_dict()`. The loop only calls `optimizer.step()`; it never calls scheduler-step separately.

**YAML hook** (per-component):

```yaml
model:
  component_optims:
    my_module:
      type: "my_optimizer"
      params:
        lr: 1.0e-4
        weight_decay: 0.01
```

## Wiring it up: the plugins list

Every new component file must be importable, and its module path must appear in the YAML's top-level `plugins:` list:

```yaml
plugins:
  - "experiment_training.components.dataloader.my_dataset"
  - "experiment_training.components.loss.my_loss"
  - "experiment_training.components.optimizer.my_optimizer"
  - "experiment_training.components.trainer.imitation_learning.my_trainer.my_trainer"
```

The strings are normal Python import paths. Order doesn't matter. Missing a plugin produces a `KeyError` from the registry at startup — see [10_troubleshooting.md](10_troubleshooting.md).

## Worked example end-to-end

A minimal new loss, registered, and used by an existing trainer. We'll use the existing [`L2LossFactory`](../experiment_training/components/loss/l2.py) as the reference shape, then walk through swapping it into a config.

The file [`experiment_training/components/loss/l2.py`](../experiment_training/components/loss/l2.py) declares:

```python
@LOSS_BUILDER_REGISTRY.register("l2_loss")
class L2LossFactory:
    def build(self, reduction: str = "sum") -> nn.Module:
        return L2Loss(reduction=reduction)
```

In a YAML, you opt into it with:

```yaml
plugins:
  - "experiment_training.components.loss.l2"
  # ...other plugins for the trainer, dataset, optimizers...

train:
  loss:
    type: "l2_loss"
    params:
      reduction: "mean"
```

At startup:

1. `load_plugins` imports `experiment_training.components.loss.l2`. The `@LOSS_BUILDER_REGISTRY.register("l2_loss")` decorator runs at import time, adding the factory to the registry.
2. `_build_loss` (in `offline_trainer.py`) calls `LOSS_BUILDER_REGISTRY.get("l2_loss")` → gets `L2LossFactory`.
3. `instantiate(L2LossFactory, _params_dict(config.train.loss.params))` instantiates the factory. Since `L2LossFactory.__init__` is the synthesized empty one (no `__init__` defined in this class), `_filter_kwargs` ignores `reduction`. *Here is where the example bends: `reduction` is consumed by `build()`, not `__init__`, which is unusual.* For your own factories, prefer to receive params in `__init__` and store them on `self`.
4. `loss_fn = factory.build()` produces the `nn.Module`. The loop moves it to `device` and passes it as `loss=` to the trainer's constructor.

If you instead build your own loss from scratch following the recipe in [Recipe: a new loss](#recipe-a-new-loss), `__init__` receives the params, `build()` takes none, and the wiring is symmetric to the other factories in the codebase.
