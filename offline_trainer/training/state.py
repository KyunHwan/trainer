"""Training state tracking."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainState:
    """Mutable training state for loops and callbacks."""

    step: int = 0
    epoch: int = 0
    max_steps: int | None = None
    max_epochs: int | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "epoch": self.epoch,
            "max_steps": self.max_steps,
            "max_epochs": self.max_epochs,
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainState":
        return cls(
            step=int(data.get("step", 0)),
            epoch=int(data.get("epoch", 0)),
            max_steps=data.get("max_steps"),
            max_epochs=data.get("max_epochs"),
            metrics=dict(data.get("metrics", {})),
        )
