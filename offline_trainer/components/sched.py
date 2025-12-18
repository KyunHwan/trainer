from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

from offline_trainer.deps.model_constructor import ConfigError, Registry, validate_kwargs

Interval = Literal["step", "epoch", "metric"]


@dataclass(frozen=True)
class SchedulerBundle:
    scheduler: Any
    interval: Interval
    frequency: int = 1
    monitor: str | None = None


@dataclass
class TorchSchedulerFactory:
    type: str
    args: list[Any] | None = None
    kwargs: dict[str, Any] | None = None
    interval: Interval = "epoch"
    frequency: int = 1
    monitor: str | None = None

    def build(self, optimizer: torch.optim.Optimizer, *, registry: Registry) -> SchedulerBundle:
        if not isinstance(self.type, str) or not self.type:
            raise ConfigError("scheduler.type must be a non-empty string", config_path=("scheduler", "type"))
        if self.args is None:
            args = []
        else:
            args = list(self.args)
        if self.kwargs is None:
            kwargs = {}
        else:
            kwargs = dict(self.kwargs)

        entry = registry.get_module(self.type, config_path=("scheduler", "type"))
        sched_cls = entry.target
        validate_kwargs(sched_cls, kwargs=kwargs, policy=entry.signature_policy, config_path=("scheduler", "kwargs"))
        try:
            sched = sched_cls(optimizer, *args, **kwargs)
        except Exception as exc:
            raise ConfigError(f"Failed to construct scheduler: {exc}", config_path=("scheduler",)) from exc
        return SchedulerBundle(scheduler=sched, interval=self.interval, frequency=self.frequency, monitor=self.monitor)

