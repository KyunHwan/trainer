"""Device helpers for nested structures."""
from __future__ import annotations

from typing import Any

import torch

from trainer.trainer.utils.tree import tree_map


def select_device(requested: str | None = None) -> torch.device:
    """Select a device based on user preference and availability."""
    if requested is None or requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def move_to_device(batch: Any, device: torch.device) -> Any:
    """Move tensors in a nested structure to a device, preserving non-tensors."""

    def _move(x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            if x.device != device:
                return x.to(device)
            else:
                return x
        return x

    return tree_map(_move, batch)


def cast_dtype(batch: Any, dtype: torch.dtype) -> Any:
    """Cast floating tensors in a nested structure to the given dtype."""

    def _cast(x: Any) -> Any:
        if isinstance(x, torch.Tensor) and x.is_floating_point():
            return x.to(dtype=dtype)
        return x

    return tree_map(_cast, batch)
