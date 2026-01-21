"""Global registries for trainer components."""
from __future__ import annotations

from trainer.templates.loss import LossFactory
from trainer.templates.optim import OptimizerFactory
from trainer.templates.dataset import DatasetFactory
from trainer.templates.trainer import Trainer

from trainer.registry.core import Registry

from typing import Any

TRAINER_REGISTRY: Registry[type[Trainer]] = Registry("trainer", expected_base=Trainer)
DATASET_BUILDER_REGISTRY: Registry[type[DatasetFactory]] = Registry("dataset_builder", expected_base=DatasetFactory)
OPTIMIZER_BUILDER_REGISTRY: Registry[type[OptimizerFactory]] = Registry(
    "optimizer_builder", expected_base=OptimizerFactory
)
LOSS_BUILDER_REGISTRY: Registry[type[LossFactory]] = Registry("loss_builder", expected_base=LossFactory)