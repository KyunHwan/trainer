"""Template for a custom DataModule."""
from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import DataLoader

from offline_trainer.data.datamodule import DataModule
from offline_trainer.registry import DATAMODULE_REGISTRY


@DATAMODULE_REGISTRY.register("my_datamodule")
@dataclass
class MyDataModule(DataModule):
    batch_size: int = 8

    def setup(self, stage: str) -> None:  # noqa: ARG002
        # Build datasets here.
        return None

    def train_dataloader(self) -> DataLoader:
        # Return your training DataLoader.
        raise NotImplementedError

    def val_dataloader(self) -> DataLoader | None:
        return None

    def test_dataloader(self) -> DataLoader | None:
        return None
