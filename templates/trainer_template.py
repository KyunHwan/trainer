"""Template for a custom trainer."""
from __future__ import annotations

from typing import Any

from torch import nn

from offline_trainer.components.callbacks import Callback
from offline_trainer.components.loggers import Logger
from offline_trainer.components.loss import Loss
from offline_trainer.components.metrics import Metrics
from offline_trainer.components.optim import OptimizerFactory
from offline_trainer.components.sched import SchedulerFactory
from offline_trainer.config.schemas import ExperimentConfig
from offline_trainer.data.datamodule import DataModule
from offline_trainer.registry import TRAINER_REGISTRY


@TRAINER_REGISTRY.register("my_trainer")
class MyTrainer:
    def __init__(
        self,
        *,
        config: ExperimentConfig,
        optimizer_factory: OptimizerFactory,
        scheduler_factory: SchedulerFactory | None,
        loss_fn: Loss,
        metrics: list[Metrics] | None,
        callbacks: list[Callback] | None,
        loggers: list[Logger] | None,
        **_: Any,
    ) -> None:
        self._config = config
        self._optimizer_factory = optimizer_factory
        self._scheduler_factory = scheduler_factory
        self._loss_fn = loss_fn
        self._metrics = metrics
        self._callbacks = callbacks
        self._loggers = loggers

    def fit(self, *, models: dict[str, nn.Module], datamodule: DataModule) -> None:
        # Implement your training logic here.
        return None
