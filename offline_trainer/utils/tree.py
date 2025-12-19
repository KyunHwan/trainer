"""Tree utilities for arbitrary nested structures."""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Callable

import torch


def tree_map(fn: Callable[[Any], Any], tree: Any, *, is_leaf: Callable[[Any], bool] | None = None) -> Any:
    """Apply fn to each leaf in a nested structure.

    Supports dict/list/tuple/namedtuple/dataclass. Other objects are treated as leaves.
    """
    if is_leaf is not None and is_leaf(tree):
        return fn(tree)

    if _is_namedtuple(tree):
        return type(tree)(*(tree_map(fn, x, is_leaf=is_leaf) for x in tree))

    if is_dataclass(tree) and not isinstance(tree, type):
        values = {f.name: tree_map(fn, getattr(tree, f.name), is_leaf=is_leaf) for f in fields(tree)}
        return type(tree)(**values)

    if isinstance(tree, dict):
        return {k: tree_map(fn, v, is_leaf=is_leaf) for k, v in tree.items()}

    if isinstance(tree, list):
        return [tree_map(fn, v, is_leaf=is_leaf) for v in tree]

    if isinstance(tree, tuple):
        return tuple(tree_map(fn, v, is_leaf=is_leaf) for v in tree)

    return fn(tree)


def tree_flatten(tree: Any, *, is_leaf: Callable[[Any], bool] | None = None) -> list[Any]:
    """Flatten leaves in a nested structure into a list."""
    leaves: list[Any] = []

    def _collect(x: Any) -> Any:
        leaves.append(x)
        return x

    tree_map(_collect, tree, is_leaf=is_leaf)
    return leaves


def _is_namedtuple(obj: Any) -> bool:
    return isinstance(obj, tuple) and hasattr(obj, "_fields")


def is_tensor_leaf(obj: Any) -> bool:
    return isinstance(obj, torch.Tensor)
