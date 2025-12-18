from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from offline_trainer.deps.model_constructor import ConfigError


@dataclass(frozen=True)
class LossOutput:
    loss: torch.Tensor
    logs: dict[str, float | torch.Tensor] = field(default_factory=dict)


class LossFn(Protocol):
    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor: ...


@dataclass
class TorchLoss:
    fn: Any
    pred_path: list[str | int] | None = None
    target_path: list[str | int] | None = None

    def __call__(self, *, outputs: Any, targets: Any, batch: Any, stage: str) -> LossOutput:
        pred = _select_tensor(outputs, self.pred_path, role="pred", config_path=("loss", "pred_path"))
        tgt = _select_tensor(targets, self.target_path, role="target", config_path=("loss", "target_path"))
        out = self.fn(pred, tgt)
        if not torch.is_tensor(out):
            raise ConfigError("loss.fn must return a torch.Tensor", config_path=("loss", "fn"))
        return LossOutput(loss=out, logs={})


def _select_tensor(obj: Any, path: list[str | int] | None, *, role: str, config_path: tuple[Any, ...]) -> torch.Tensor:
    if path is not None:
        try:
            obj = _extract(obj, path)
        except Exception as exc:
            raise ConfigError(f"Failed to extract {role} via path: {exc}", config_path=config_path) from exc

    obj = _unwrap_single(obj, role=role, config_path=config_path)
    if not torch.is_tensor(obj):
        raise ConfigError(f"{role} must be a torch.Tensor (got {type(obj).__name__})", config_path=config_path)
    return obj


def _unwrap_single(obj: Any, *, role: str, config_path: tuple[Any, ...]) -> Any:
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, dict):
        if len(obj) == 1:
            return next(iter(obj.values()))
        raise ConfigError(f"Ambiguous {role}: mapping has {len(obj)} keys; set {config_path[-1]} explicitly", config_path=config_path)
    if isinstance(obj, (list, tuple)):
        if len(obj) == 1:
            return obj[0]
        raise ConfigError(f"Ambiguous {role}: sequence has {len(obj)} items; set {config_path[-1]} explicitly", config_path=config_path)
    return obj


def _extract(obj: Any, path: list[str | int]) -> Any:
    cur = obj
    for seg in path:
        if isinstance(seg, str):
            if not isinstance(cur, dict) or seg not in cur:
                raise KeyError(seg)
            cur = cur[seg]
        elif isinstance(seg, int):
            if not isinstance(cur, (list, tuple)) or seg < 0 or seg >= len(cur):
                raise IndexError(seg)
            cur = cur[seg]
        else:
            raise TypeError(f"Invalid path segment type: {type(seg).__name__}")
    return cur

