"""Example custom trainer registered via plugin import."""
from __future__ import annotations

from offline_trainer.registry import TRAINER_REGISTRY
from offline_trainer.training.trainer import DefaultTrainer


@TRAINER_REGISTRY.register("custom_trainer_v1")
class CustomTrainer(DefaultTrainer):
    """Thin wrapper around DefaultTrainer to demonstrate plugins."""

    pass
