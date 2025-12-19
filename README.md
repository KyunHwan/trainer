# Offline Trainer

Offline Trainer is a lightweight, config-driven training loop that composes models, data, and training components from registries. It relies on policy_constructor for model construction and keeps the training stack modular so you can swap components via YAML.

## Quick start
- Run a minimal smoke test:
  python examples/run_from_python.py
- Or call the API from your own script:
  from offline_trainer.api import train
  train("examples/configs/minimal.yaml")

## Configuration overview
The entrypoint is `offline_trainer.api.train`, which loads and validates a YAML config.

Minimal shape:
```yaml
plugins: []
seed: 123
deterministic: false
device: "auto"
model:
  config_path: "policy_constructor/configs/examples/sequential_mlp.yaml"
  # or: config: { ... }  # inline policy_constructor config dict
data:
  datamodule:
    type: "random_regression"
    params: {}
train:
  model_input: "x"
  trainer: { type: "default_trainer", params: {} }
  optimizer: { type: "adamw", params: { lr: 0.001 } }
  scheduler: { type: "none", params: {} }
  loss: { type: "mse", params: { target_key: "y" } }
  metrics: []
  callbacks: []
  loggers: [{ type: "stdout", params: { prefix: "train" } }]
  max_epochs: 1
  max_steps: null
  accumulate_grad_batches: 1
  amp: false
  gradient_clip_val: null
  log_every_n_steps: 1
  ema: { enabled: false, decay: 0.999 }
  checkpoint:
    save_dir: "runs"
    save_every_n_steps: null
    save_last: true
    resume_from: null
```

Notes:
- `plugins` is a list of importable modules that register custom components.
- `model_input` is used by the training loop to select the model input from the batch.
- The YAML loader supports `defaults` composition and deep merge.

## Comprehensive custom components example
This example shows where to put custom components, how to register them, and how to use them from YAML.

1) Create a module with custom components (any importable module works). For a quick local example, add `examples/extensions/my_components.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from offline_trainer.data.utils import worker_init_fn
from offline_trainer.registry import (
    CALLBACK_REGISTRY,
    DATAMODULE_REGISTRY,
    LOGGER_REGISTRY,
    LOSS_REGISTRY,
    METRICS_REGISTRY,
    OPTIMIZER_REGISTRY,
    SCHEDULER_REGISTRY,
    TRAINER_REGISTRY,
)
from offline_trainer.training.state import TrainState
from offline_trainer.training.trainer import DefaultTrainer
from offline_trainer.utils.selection import select


class _ToyDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self._x = x
        self._y = y

    def __len__(self) -> int:
        return self._x.size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"x": self._x[idx], "y": self._y[idx]}


@DATAMODULE_REGISTRY.register("toy_regression_v1")
@dataclass
class ToyRegressionDataModule:
    batch_size: int = 8
    num_workers: int = 0
    dataset_size: int = 64
    input_dim: int = 8
    output_dim: int = 16
    seed: int = 123

    def setup(self, stage: str) -> None:  # noqa: ARG002
        gen = torch.Generator().manual_seed(self.seed)
        x = torch.randn(self.dataset_size, self.input_dim, generator=gen)
        y = torch.randn(self.dataset_size, self.output_dim, generator=gen)
        self._train_ds = _ToyDataset(x, y)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn(),
        )

    def val_dataloader(self) -> DataLoader | None:
        return None

    def test_dataloader(self) -> DataLoader | None:
        return None


@LOSS_REGISTRY.register("smooth_l1")
@dataclass
class SmoothL1Loss:
    pred_key: str | int | None = None
    target_key: str | int | None = "y"
    beta: float = 1.0

    def __call__(self, batch: Any, outputs: Any) -> torch.Tensor:
        preds = select(outputs, self.pred_key)
        target = select(batch, self.target_key)
        return torch.nn.functional.smooth_l1_loss(preds, target, beta=self.beta)


@METRICS_REGISTRY.register("mae")
@dataclass
class MeanAbsoluteError:
    pred_key: str | int | None = None
    target_key: str | int | None = "y"
    _sum: float = 0.0
    _count: int = 0

    def reset(self) -> None:
        self._sum = 0.0
        self._count = 0

    def update(self, batch: Any, outputs: Any) -> None:
        preds = select(outputs, self.pred_key)
        target = select(batch, self.target_key)
        mae = torch.mean(torch.abs(preds - target)).item()
        self._sum += float(mae)
        self._count += 1

    def compute(self) -> dict[str, float]:
        if self._count == 0:
            return {"mae": 0.0}
        return {"mae": self._sum / self._count}


@CALLBACK_REGISTRY.register("print_every_n")
@dataclass
class PrintEveryN:
    every_n: int = 10

    def on_train_start(self, state: TrainState) -> None:  # noqa: ARG002
        return None

    def on_step_end(self, state: TrainState) -> None:
        if self.every_n > 0 and state.step % self.every_n == 0:
            print(f"[callback] step={state.step} metrics={state.metrics}")

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:  # noqa: ARG002
        return None


@LOGGER_REGISTRY.register("simple_stdout")
@dataclass
class SimpleStdoutLogger:
    prefix: str = "train"

    def log(self, metrics: dict[str, Any], step: int) -> None:
        parts = ", ".join(
            f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        )
        print(f"[{self.prefix}] step={step} {parts}")

    def flush(self) -> None:
        return None


@OPTIMIZER_REGISTRY.register("adamw_eps")
@dataclass
class AdamWEpsFactory:
    lr: float = 5e-4
    weight_decay: float = 0.01
    eps: float = 1e-8

    def build(self, params: Any) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            eps=self.eps,
        )


@SCHEDULER_REGISTRY.register("linear_warmup")
@dataclass
class LinearWarmupFactory:
    warmup_steps: int = 100

    def build(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LambdaLR:
        def lr_lambda(step: int) -> float:
            if self.warmup_steps <= 0:
                return 1.0
            return min((step + 1) / self.warmup_steps, 1.0)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


@TRAINER_REGISTRY.register("default_trainer_plus")
class DefaultTrainerPlus(DefaultTrainer):
    def fit(self, *, models: dict[str, torch.nn.Module], datamodule: Any) -> None:
        print("starting training")
        super().fit(models=models, datamodule=datamodule)
```

2) Point your YAML at the plugin module and the registry keys:

```yaml
plugins:
  - "examples.extensions.my_components"
model:
  config_path: "policy_constructor/configs/examples/sequential_mlp.yaml"
data:
  datamodule:
    type: "toy_regression_v1"
    params:
      batch_size: 4
      dataset_size: 16
train:
  model_input: "x"
  trainer:
    type: "default_trainer_plus"
    params: {}
  optimizer:
    type: "adamw_eps"
    params:
      lr: 0.0005
  scheduler:
    type: "linear_warmup"
    params:
      warmup_steps: 5
  loss:
    type: "smooth_l1"
    params:
      target_key: "y"
  metrics:
    - type: "mae"
      params:
        target_key: "y"
  callbacks:
    - type: "print_every_n"
      params:
        every_n: 1
  loggers:
    - type: "simple_stdout"
      params:
        prefix: "train"
  max_steps: 5
  max_epochs: 1
```

3) Run it:

```
python -c "from offline_trainer.api import train; train('path/to/your.yaml')"
```

Notes:
- Use `templates/` for starter component skeletons.
- For models, supply a policy_constructor config via `model.config_path` or inline `model.config`.

## Project layout
- `offline_trainer/`: core library and training loop.
- `examples/`: sample configs and plugin modules.
- `templates/`: copy-paste component templates.
- `tests/`: pytest suite.
- `policy_constructor/`: model construction library used by `offline_trainer`.
