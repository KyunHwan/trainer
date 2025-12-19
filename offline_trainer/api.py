"""Public API entrypoint for training."""
from __future__ import annotations

import os

from offline_trainer.config.loader import load_config
from offline_trainer.config.schemas import ExperimentConfig, validate_config
from offline_trainer.modeling.factories import PolicyConstructorModelFactory
from offline_trainer.registry import (
    CALLBACK_REGISTRY,
    DATAMODULE_REGISTRY,
    LOGGER_REGISTRY,
    LOSS_REGISTRY,
    METRICS_REGISTRY,
    OPTIMIZER_REGISTRY,
    SCHEDULER_REGISTRY,
    TRAINER_REGISTRY,
    register_builtins,
)
from offline_trainer.registry.plugins import load_plugins
from offline_trainer.utils.import_utils import instantiate


def _params_dict(params) -> dict:
    if hasattr(params, "model_dump"):
        return params.model_dump()
    return params


def train(config_path: str) -> None:
    """Train an experiment specified entirely by YAML config."""
    raw = load_config(config_path)
    config: ExperimentConfig = validate_config(raw)

    register_builtins()
    load_plugins(config.plugins)

    if config.model.config_path and not os.path.isabs(config.model.config_path):
        config.model.config_path = os.path.abspath(
            os.path.join(os.path.dirname(config_path), config.model.config_path)
        )

    model_factory = PolicyConstructorModelFactory()
    model = model_factory.build(config.model.as_dict())

    # this allows for multiple models to be trained (ex. in offline rl setting)
    models = {"main": model} if not isinstance(model, dict) else model

    datamodule_cls = DATAMODULE_REGISTRY.get(config.data.datamodule.type)
    datamodule = instantiate(datamodule_cls, config.data.datamodule.params, config=config)

    optimizer_cls = OPTIMIZER_REGISTRY.get(config.train.optimizer.type)
    optimizer_factory = instantiate(optimizer_cls, _params_dict(config.train.optimizer.params))

    scheduler_cls = SCHEDULER_REGISTRY.get(config.train.scheduler.type)
    scheduler_factory = instantiate(scheduler_cls, _params_dict(config.train.scheduler.params))

    loss_cls = LOSS_REGISTRY.get(config.train.loss.type)
    loss_fn = instantiate(loss_cls, _params_dict(config.train.loss.params))

    metrics = [
        instantiate(METRICS_REGISTRY.get(spec.type), _params_dict(spec.params))
        for spec in config.train.metrics
    ]
    callbacks = [
        instantiate(CALLBACK_REGISTRY.get(spec.type), _params_dict(spec.params))
        for spec in config.train.callbacks
    ]
    loggers = [
        instantiate(LOGGER_REGISTRY.get(spec.type), _params_dict(spec.params))
        for spec in config.train.loggers
    ]

    trainer_cls = TRAINER_REGISTRY.get(config.train.trainer.type)
    trainer = instantiate(
        trainer_cls,
        _params_dict(config.train.trainer.params),
        config=config,
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
        loss_fn=loss_fn,
        metrics=metrics,
        callbacks=callbacks,
        loggers=loggers,
    )

    trainer.fit(models=models, datamodule=datamodule)
