from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from offline_trainer.deps.model_constructor import ConfigError, Registry, validate_kwargs


@dataclass
class TorchOptimizerFactory:
    type: str
    args: list[Any] | None = None
    kwargs: dict[str, Any] | None = None

    def build(self, model: torch.nn.Module, *, registry: Registry) -> torch.optim.Optimizer:
        if not isinstance(self.type, str) or not self.type:
            raise ConfigError("optimizer.type must be a non-empty string", config_path=("optimizer", "type"))
        if self.args is None:
            args = []
        else:
            args = list(self.args)
        if self.kwargs is None:
            kwargs = {}
        else:
            kwargs = dict(self.kwargs)

        entry = registry.get_module(self.type, config_path=("optimizer", "type"))
        optim_cls = entry.target
        validate_kwargs(optim_cls, kwargs=kwargs, policy=entry.signature_policy, config_path=("optimizer", "kwargs"))
        try:
            return optim_cls(model.parameters(), *args, **kwargs)
        except Exception as exc:
            raise ConfigError(f"Failed to construct optimizer: {exc}", config_path=("optimizer",)) from exc

