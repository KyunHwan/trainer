"""Metrics interface and built-ins."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch

from offline_trainer.utils.selection import select


@runtime_checkable
class Metrics(Protocol):
    """Aggregates metrics over training steps."""

    def reset(self) -> None: ...

    def update(self, batch: Any, outputs: Any) -> None: ...

    def compute(self) -> dict[str, float]: ...


@dataclass
class MeanSquaredError:
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
        mse = torch.mean((preds - target) ** 2).item()
        self._sum += float(mse)
        self._count += 1

    def compute(self) -> dict[str, float]:
        if self._count == 0:
            return {"mse": 0.0}
        return {"mse": self._sum / self._count}


@dataclass
class NoOpMetrics:
    _state: dict[str, float] = field(default_factory=dict)

    def reset(self) -> None:
        self._state.clear()

    def update(self, batch: Any, outputs: Any) -> None:  # noqa: ARG002
        return None

    def compute(self) -> dict[str, float]:
        return dict(self._state)
