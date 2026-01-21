"""Optimizer factory interface and built-ins."""
from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class OptimizerFactory(Protocol):
    """ Custom Optimizer Factory """

    def build(self, params: Iterable[nn.Parameter], **kwargs) -> torch.optim.Optimizer: ...