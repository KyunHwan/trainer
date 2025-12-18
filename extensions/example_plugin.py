from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch


@dataclass
class ConstantRegressionDataModule:
    batch_size: int = 16
    steps_per_epoch: int = 10
    x_shape: list[int] = None  # type: ignore[assignment]
    y_shape: list[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.x_shape is None:
            self.x_shape = [32]
        if self.y_shape is None:
            self.y_shape = [16]

    def setup(self, stage: str | None = None) -> None:
        return

    def train_dataloader(self) -> Iterable[Any]:
        x = torch.zeros(self.batch_size, *self.x_shape)
        y = torch.zeros(self.batch_size, *self.y_shape)
        for _ in range(self.steps_per_epoch):
            yield {"x": x.clone(), "y": y.clone()}

    def val_dataloader(self) -> Iterable[Any] | None:
        return None


@dataclass
class VerboseTrainer:
    max_epochs: int = 1
    log_every_n_steps: int = 5

    def fit(self, **kwargs: Any) -> None:
        print("Running VerboseTrainer")
        default_trainer = kwargs["registry"].get_module("trainer.default").target(
            max_epochs=self.max_epochs,
            log_every_n_steps=self.log_every_n_steps,
        )
        default_trainer.fit(**kwargs)


def register(registry) -> None:
    registry.register_module("data.constant_regression", ConstantRegressionDataModule, signature_policy="strict", tags=("data", "ext"))
    registry.register_module("trainer.verbose", VerboseTrainer, signature_policy="strict", tags=("trainer", "ext"))

