"""Trainer interface and default implementation."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from torch import nn

from offline_trainer.components.callbacks import Callback
from offline_trainer.components.loggers import Logger
from offline_trainer.components.loss import Loss
from offline_trainer.components.metrics import Metrics
from offline_trainer.components.optim import OptimizerFactory
from offline_trainer.components.sched import SchedulerFactory
from offline_trainer.config.schemas import ExperimentConfig
from offline_trainer.data.datamodule import DataModule
from offline_trainer.training.loop import fit_loop


@runtime_checkable
class Trainer(Protocol):
    """Trainer interface."""

    def fit(self, *, models: dict[str, nn.Module], datamodule: DataModule) -> None: ...


class DefaultTrainer:
    """Default training orchestration with configurable components."""

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        optimizer_factory: OptimizerFactory,
        scheduler_factory: SchedulerFactory | None,
        loss_fn: Loss,
        metrics: list[Metrics],
        callbacks: list[Callback],
        loggers: list[Logger],
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
        fit_loop(
            models=models,
            datamodule=datamodule,
            config=self._config,
            optimizer_factory=self._optimizer_factory,
            scheduler_factory=self._scheduler_factory,
            loss_fn=self._loss_fn,
            metrics=self._metrics,
            callbacks=self._callbacks,
            loggers=self._loggers,
        )
