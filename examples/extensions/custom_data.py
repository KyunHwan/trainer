"""Example custom datamodule registered via plugin import."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset

from offline_trainer.data.utils import worker_init_fn
from offline_trainer.registry import DATAMODULE_REGISTRY


class _TinyDataset(Dataset):
    def __init__(self, num_samples: int, input_dim: int, output_dim: int, seed: int) -> None:
        gen = torch.Generator().manual_seed(seed)
        self._x = torch.randn(num_samples, input_dim, generator=gen)
        self._y = torch.randn(num_samples, output_dim, generator=gen)

    def __len__(self) -> int:
        return self._x.size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"x": self._x[idx], "y": self._y[idx]}


@DATAMODULE_REGISTRY.register("custom_data_v1")
@dataclass
class CustomDataModule:
    batch_size: int = 4
    num_workers: int = 0
    dataset_size: int = 16
    input_dim: int = 8
    output_dim: int = 16
    seed: int = 123

    def setup(self, stage: str) -> None:  # noqa: ARG002
        self._train_ds = _TinyDataset(
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
