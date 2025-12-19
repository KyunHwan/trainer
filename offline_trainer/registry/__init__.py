"""Global registries for offline_trainer components."""
from __future__ import annotations

from offline_trainer.components.callbacks import Callback
from offline_trainer.components.loggers import Logger
from offline_trainer.components.loss import Loss
from offline_trainer.components.metrics import Metrics
from offline_trainer.components.optim import OptimizerFactory
from offline_trainer.components.sched import SchedulerFactory
from offline_trainer.data.datamodule import DataModule
from offline_trainer.registry.core import Registry
from offline_trainer.training.trainer import Trainer

TRAINER_REGISTRY: Registry[type[Trainer]] = Registry("trainer", expected_base=Trainer)
DATAMODULE_REGISTRY: Registry[type[DataModule]] = Registry("datamodule", expected_base=DataModule)
OPTIMIZER_REGISTRY: Registry[type[OptimizerFactory]] = Registry(
    "optimizer", expected_base=OptimizerFactory
)
SCHEDULER_REGISTRY: Registry[type[SchedulerFactory]] = Registry(
    "scheduler", expected_base=SchedulerFactory
)
LOSS_REGISTRY: Registry[type[Loss]] = Registry("loss", expected_base=Loss)
METRICS_REGISTRY: Registry[type[Metrics]] = Registry("metrics", expected_base=Metrics)
CALLBACK_REGISTRY: Registry[type[Callback]] = Registry("callback", expected_base=Callback)
LOGGER_REGISTRY: Registry[type[Logger]] = Registry("logger", expected_base=Logger)


def register_builtins() -> None:
    """Register built-in components into registries."""
    from offline_trainer.registry.builtins import register_builtins as _register

    _register()
