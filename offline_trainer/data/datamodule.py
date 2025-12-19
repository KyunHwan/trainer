"""DataModule protocol and built-in implementations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch.utils.data import DataLoader, Dataset

from offline_trainer.data.utils import worker_init_fn


@runtime_checkable
class DataModule(Protocol):
    """Minimal data module interface."""

    def setup(self, stage: str) -> None: ...

    def train_dataloader(self) -> DataLoader: ...

    def val_dataloader(self) -> DataLoader | None: ...

    def test_dataloader(self) -> DataLoader | None: ...


class _RandomRegressionDataset(Dataset):
    def __init__(self, num_samples: int, input_dim: int, output_dim: int, seed: int) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self._x = torch.randn(num_samples, input_dim, generator=gen)
        self._y = torch.randn(num_samples, output_dim, generator=gen)

    def __len__(self) -> int:
        return self._x.size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"x": self._x[idx], "y": self._y[idx]}


@dataclass
class RandomRegressionDataModule:
    """Simple in-memory regression data for smoke tests."""

    batch_size: int = 8
    num_workers: int = 0
    dataset_size: int = 64
    input_dim: int = 8
    output_dim: int = 16
    seed: int = 0

    def setup(self, stage: str) -> None:
        self._train_ds = _RandomRegressionDataset(
            num_samples=self.dataset_size,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            seed=self.seed,
        )

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
