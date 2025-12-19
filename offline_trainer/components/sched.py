"""Scheduler factory interface and built-ins."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class SchedulerFactory(Protocol):
    """Builds schedulers for an optimizer."""

    def build(self, optimizer: torch.optim.Optimizer) -> Any: ...


@dataclass
class NoneSchedulerFactory:
    def build(self, optimizer: torch.optim.Optimizer) -> None:  # noqa: ARG002
        return None


@dataclass
class StepLRFactory:
    step_size: int = 10
    gamma: float = 0.1

    def build(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.StepLR:
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.gamma
        )


@dataclass
class CosineAnnealingLRFactory:
    T_max: int = 50
    eta_min: float = 0.0

    def build(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.CosineAnnealingLR:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.T_max, eta_min=self.eta_min
        )
