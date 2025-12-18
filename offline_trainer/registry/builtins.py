from __future__ import annotations

import torch

from offline_trainer.deps.model_constructor import Registry

from offline_trainer.components.callbacks import ModelCheckpoint
from offline_trainer.components.data import RandomRegressionDataModule
from offline_trainer.components.io import MappingIO
from offline_trainer.components.loggers import StdoutLogger
from offline_trainer.components.loss import TorchLoss
from offline_trainer.components.optim import TorchOptimizerFactory
from offline_trainer.components.sched import TorchSchedulerFactory
from offline_trainer.engine.trainer import DefaultTrainer


def register_offline_trainer_builtins(registry: Registry) -> None:
    registry.register_module("trainer.default", DefaultTrainer, signature_policy="strict", tags=("trainer",))

    registry.register_module(
        "data.random_regression",
        RandomRegressionDataModule,
        signature_policy="strict",
        tags=("data", "builtin"),
    )
    registry.register_module("io.mapping", MappingIO, signature_policy="strict", tags=("io", "builtin"))
    registry.register_module("loss.torch", TorchLoss, signature_policy="best_effort", tags=("loss", "builtin"))

    registry.register_module(
        "optim_factory.torch",
        TorchOptimizerFactory,
        signature_policy="strict",
        tags=("optim", "factory", "builtin"),
    )
    registry.register_module(
        "sched_factory.torch",
        TorchSchedulerFactory,
        signature_policy="strict",
        tags=("sched", "factory", "builtin"),
    )

    registry.register_module("cb.model_checkpoint", ModelCheckpoint, signature_policy="strict", tags=("callback", "builtin"))
    registry.register_module("log.stdout", StdoutLogger, signature_policy="strict", tags=("logger", "builtin"))

    _register_torch_nn_loss_modules(registry)
    _register_torch_optimizer_classes(registry)
    _register_torch_scheduler_classes(registry)


def _register_torch_optimizer_classes(registry: Registry) -> None:
    optim = torch.optim
    for name in [
        "AdamW",
        "Adam",
        "SGD",
        "RMSprop",
        "Adagrad",
        "Adamax",
        "NAdam",
    ]:
        if hasattr(optim, name):
            registry.register_module(
                f"optim_cls.{name}",
                getattr(optim, name),
                signature_policy="strict",
                tags=("torch.optim", "optim_cls"),
            )


def _register_torch_nn_loss_modules(registry: Registry) -> None:
    nn = torch.nn
    existing = set(registry.list_modules())
    for name in [
        "MSELoss",
        "L1Loss",
        "SmoothL1Loss",
        "HuberLoss",
        "CrossEntropyLoss",
        "NLLLoss",
        "BCEWithLogitsLoss",
        "BCELoss",
        "KLDivLoss",
    ]:
        key = f"nn.{name}"
        if key in existing:
            continue
        if hasattr(nn, name):
            registry.register_module(
                key,
                getattr(nn, name),
                signature_policy="strict",
                tags=("torch.nn", "loss"),
            )


def _register_torch_scheduler_classes(registry: Registry) -> None:
    sched = torch.optim.lr_scheduler
    for name in [
        "StepLR",
        "CosineAnnealingLR",
        "ExponentialLR",
        "ReduceLROnPlateau",
        "OneCycleLR",
        "LinearLR",
        "MultiStepLR",
    ]:
        if hasattr(sched, name):
            registry.register_module(
                f"sched_cls.{name}",
                getattr(sched, name),
                signature_policy="strict",
                tags=("torch.optim.lr_scheduler", "sched_cls"),
            )
