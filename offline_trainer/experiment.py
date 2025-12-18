from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from offline_trainer.config.errors import enrich_config_error
from offline_trainer.config.validate import RunConfig, validate_post_model, validate_preflight
from offline_trainer.deps.model_constructor import ConfigError, Registry, build_model, instantiate_value, resolve_config
from offline_trainer.engine.seed import seed_everything
from offline_trainer.registry import build_registry


def run_experiment(config_path: str | Path) -> None:
    p = Path(config_path)
    resolved = resolve_config(p)
    cfg = resolved.data
    sm = resolved.source_map

    run = validate_preflight(cfg, source_map=sm)
    seed_everything(run.seed, deterministic=run.deterministic, include_cuda=(run.device.type == "cuda"))

    registry = build_registry()
    model = _build_model_with_wrapped_errors(p, registry=registry, source_map=sm)

    validate_post_model(cfg, registry=registry, source_map=sm, model=model)

    datamodule = _instantiate(cfg["data"], registry=registry, settings=resolved.settings, source_map=sm, config_path=("data",))
    io = _instantiate(cfg["io"], registry=registry, settings=resolved.settings, source_map=sm, config_path=("io",))
    loss = _instantiate(cfg["loss"], registry=registry, settings=resolved.settings, source_map=sm, config_path=("loss",))

    optimizer_factory = _instantiate(cfg["optimizer"], registry=registry, settings=resolved.settings, source_map=sm, config_path=("optimizer",))
    scheduler_factory = None
    if cfg.get("scheduler") is not None:
        scheduler_factory = _instantiate(
            cfg["scheduler"],
            registry=registry,
            settings=resolved.settings,
            source_map=sm,
            config_path=("scheduler",),
        )

    metrics_raw = cfg.get("metrics") or []
    callbacks_raw = cfg.get("callbacks") or []
    loggers_raw = cfg.get("loggers") or []

    metrics = [
        _instantiate(item, registry=registry, settings=resolved.settings, source_map=sm, config_path=("metrics", i))
        for i, item in enumerate(metrics_raw)
    ]
    callbacks = [
        _instantiate(item, registry=registry, settings=resolved.settings, source_map=sm, config_path=("callbacks", i))
        for i, item in enumerate(callbacks_raw)
    ]
    loggers = [
        _instantiate(item, registry=registry, settings=resolved.settings, source_map=sm, config_path=("loggers", i))
        for i, item in enumerate(loggers_raw)
    ]

    trainer = _instantiate(cfg["trainer"], registry=registry, settings=resolved.settings, source_map=sm, config_path=("trainer",))

    trainer.fit(
        model=model,
        datamodule=datamodule,
        io=io,
        loss=loss,
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
        metrics=metrics,
        callbacks=callbacks,
        loggers=loggers,
        registry=registry,
        run=run,
    )


def _instantiate(obj: Any, *, registry: Registry, settings: Any, source_map: Any, config_path: tuple[Any, ...]) -> Any:
    try:
        return instantiate_value(
            obj,
            registry=registry,
            settings=settings,
            source_map=source_map,
            config_path=config_path,
        )
    except ConfigError as exc:
        raise enrich_config_error(exc, source_map=source_map) from exc


def _build_model_with_wrapped_errors(path: Path, *, registry: Registry, source_map: Any) -> torch.nn.Module:
    try:
        return build_model(path, registry=registry)
    except ConfigError as exc:
        raise enrich_config_error(exc, source_map=source_map) from exc
    except ValueError as exc:
        msg = str(exc)
        raise ConfigError(
            f"Registry error during imports/model build: {msg}",
            config_path=("imports",),
            location=source_map.get(("imports",)),
            suggestions=[
                "Ensure imports are unique (no duplicates after defaults merge)",
                "Ensure each plugin register(registry) uses globally unique keys",
                "Ensure plugin register(registry) is idempotent",
            ],
        ) from exc
