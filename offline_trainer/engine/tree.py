from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, Callable, TypeVar

import torch

T = TypeVar("T")


def tree_map(fn: Callable[[torch.Tensor], torch.Tensor], obj: Any) -> Any:
    if torch.is_tensor(obj):
        return fn(obj)

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        values = {f.name: tree_map(fn, getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        return obj.__class__(**values)

    if isinstance(obj, Mapping):
        mapped = {k: tree_map(fn, v) for k, v in obj.items()}
        if isinstance(obj, dict):
            return mapped
        try:
            return obj.__class__(mapped)
        except Exception:
            return mapped

    if isinstance(obj, tuple) and hasattr(obj, "_fields"):  # namedtuple
        return obj.__class__(*(tree_map(fn, v) for v in obj))

    if isinstance(obj, tuple):
        return tuple(tree_map(fn, v) for v in obj)

    if isinstance(obj, list):
        return [tree_map(fn, v) for v in obj]

    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return [tree_map(fn, v) for v in obj]

    return obj


def move_to_device(obj: Any, device: torch.device, *, non_blocking: bool = True) -> Any:
    def _move(t: torch.Tensor) -> torch.Tensor:
        return t.to(device=device, non_blocking=non_blocking)

    return tree_map(_move, obj)


def to_float(x: float | torch.Tensor) -> float:
    if isinstance(x, float):
        return x
    return float(x.detach().cpu().item())

