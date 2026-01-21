"""Loss interface and built-ins."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn as nn


@runtime_checkable
class LossFactory(Protocol):
    """ Custom Loss Factory """

    def build(self) -> nn.Module: ...