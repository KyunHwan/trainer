"""DataModule protocol and built-in implementations."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch.utils.data import Dataset



@runtime_checkable
class DatasetFactory(Protocol):
    """ Dataset Factory """

    def build(self) -> Dataset: ...