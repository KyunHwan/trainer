"""Loss interface and built-ins."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch
from torch.nn import functional as F

from offline_trainer.utils.selection import select


@runtime_checkable
class Loss(Protocol):
    """Computes a scalar loss from batch and model outputs."""

    def __call__(self, batch: Any, outputs: Any) -> torch.Tensor: ...


@dataclass
class MSELoss:
    pred_key: str | int | None = None
    target_key: str | int | None = "y"

    def __call__(self, batch: Any, outputs: Any) -> torch.Tensor:
        preds = select(outputs, self.pred_key)
        target = select(batch, self.target_key)
        return F.mse_loss(preds, target)


@dataclass
class CrossEntropyLoss:
    logits_key: str | int | None = None
    target_key: str | int | None = "y"

    def __call__(self, batch: Any, outputs: Any) -> torch.Tensor:
        logits = select(outputs, self.logits_key)
        target = select(batch, self.target_key)
        return F.cross_entropy(logits, target)
