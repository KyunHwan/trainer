"""Register built-in components."""
from __future__ import annotations

from offline_trainer.components.callbacks import NoOpCallback
from offline_trainer.components.loggers import NoOpLogger, StdoutLogger
from offline_trainer.components.loss import CrossEntropyLoss, MSELoss
from offline_trainer.components.metrics import MeanSquaredError, NoOpMetrics
from offline_trainer.components.optim import AdamWFactory, SGDFactory
from offline_trainer.components.sched import CosineAnnealingLRFactory, NoneSchedulerFactory, StepLRFactory
from offline_trainer.data.datamodule import RandomRegressionDataModule
from offline_trainer.registry import (
    CALLBACK_REGISTRY,
    DATAMODULE_REGISTRY,
    LOGGER_REGISTRY,
    LOSS_REGISTRY,
    METRICS_REGISTRY,
    OPTIMIZER_REGISTRY,
    SCHEDULER_REGISTRY,
    TRAINER_REGISTRY,
)
from offline_trainer.training.trainer import DefaultTrainer


def register_builtins() -> None:
    """Register built-in trainers, data, and components."""
    _add_if_missing(TRAINER_REGISTRY, "default_trainer", DefaultTrainer)
    _add_if_missing(DATAMODULE_REGISTRY, "random_regression", RandomRegressionDataModule)

    _add_if_missing(OPTIMIZER_REGISTRY, "adamw", AdamWFactory)
    _add_if_missing(OPTIMIZER_REGISTRY, "sgd", SGDFactory)

    _add_if_missing(SCHEDULER_REGISTRY, "none", NoneSchedulerFactory)
    _add_if_missing(SCHEDULER_REGISTRY, "step_lr", StepLRFactory)
    _add_if_missing(SCHEDULER_REGISTRY, "cosine", CosineAnnealingLRFactory)

    _add_if_missing(LOSS_REGISTRY, "mse", MSELoss)
    _add_if_missing(LOSS_REGISTRY, "cross_entropy", CrossEntropyLoss)

    _add_if_missing(METRICS_REGISTRY, "mse", MeanSquaredError)
    _add_if_missing(METRICS_REGISTRY, "noop", NoOpMetrics)

    _add_if_missing(CALLBACK_REGISTRY, "noop", NoOpCallback)

    _add_if_missing(LOGGER_REGISTRY, "noop", NoOpLogger)
    _add_if_missing(LOGGER_REGISTRY, "stdout", StdoutLogger)


def _add_if_missing(registry, key: str, obj) -> None:
    if not registry.has(key):
        registry.add(key, obj)
