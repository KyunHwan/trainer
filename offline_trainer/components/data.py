from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch


@dataclass
class RandomRegressionDataModule:
    batch_size: int = 32
    steps_per_epoch: int = 100
    val_steps_per_epoch: int = 0
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
        for _ in range(self.steps_per_epoch):
            x = torch.randn(self.batch_size, *self.x_shape)
            y = torch.randn(self.batch_size, *self.y_shape)
            yield {"x": x, "y": y}

    def val_dataloader(self) -> Iterable[Any] | None:
        if self.val_steps_per_epoch <= 0:
            return None
        for _ in range(self.val_steps_per_epoch):
            x = torch.randn(self.batch_size, *self.x_shape)
            y = torch.randn(self.batch_size, *self.y_shape)
            yield {"x": x, "y": y}

