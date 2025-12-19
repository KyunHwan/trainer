"""Optimizer factory interface and built-ins."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class OptimizerFactory(Protocol):
    """Builds optimizers for a parameter iterable."""

    def build(self, params: Iterable[nn.Parameter]) -> torch.optim.Optimizer: ...


@dataclass
class AdamWFactory:
    lr: float = 1e-3
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)

    def build(self, params: Iterable[nn.Parameter]) -> torch.optim.Optimizer:
        return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay, betas=self.betas)


@dataclass
class SGDFactory:
    lr: float = 1e-2
    momentum: float = 0.0
    weight_decay: float = 0.0

    def build(self, params: Iterable[nn.Parameter]) -> torch.optim.Optimizer:
        return torch.optim.SGD(
            params, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay
        )
