"""Global registries for offline_trainer components."""
from __future__ import annotations

from offline_trainer.templates.loss import LossFactory
from offline_trainer.templates.optim import OptimizerFactory
from offline_trainer.templates.dataset import DatasetFactory
from offline_trainer.templates.trainer import Trainer

from offline_trainer.registry.core import Registry

from typing import Any

TRAINER_REGISTRY: Registry[type[Trainer]] = Registry("trainer", expected_base=Trainer)
DATASET_BUILDER_REGISTRY: Registry[type[DatasetFactory]] = Registry("dataset_builder", expected_base=DatasetFactory)
OPTIMIZER_BUILDER_REGISTRY: Registry[type[OptimizerFactory]] = Registry(
    "optimizer_builder", expected_base=OptimizerFactory
)
LOSS_BUILDER_REGISTRY: Registry[type[LossFactory]] = Registry("loss_builder", expected_base=LossFactory)