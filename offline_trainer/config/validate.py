from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Literal

import torch

from offline_trainer.config.errors import raise_config_error
from offline_trainer.deps.model_constructor import ConfigError, Registry, SourceMap

Precision = Literal["fp32", "fp16-mixed", "bf16-mixed"]


@dataclass(frozen=True)
class RunConfig:
    name: str
    out_dir: str
    seed: int
    device: torch.device
    precision: Precision
    deterministic: bool
    resume_from: str | None = None


def validate_preflight(config: Any, *, source_map: SourceMap) -> RunConfig:
    if not isinstance(config, dict):
        raise_config_error("config must be a mapping", config_path=(), source_map=source_map)

    _validate_unique_imports(config.get("imports", []), source_map=source_map)

    run_raw = _require_mapping(config.get("run"), ("run",), source_map=source_map, required=True)
    run = _parse_run(run_raw, source_map=source_map)

    _require_spec(config.get("trainer"), ("trainer",), source_map=source_map, expected_prefix="trainer.")
    _require_spec(config.get("data"), ("data",), source_map=source_map, expected_prefix="data.")
    _require_spec(config.get("io"), ("io",), source_map=source_map, expected_prefix="io.")
    _validate_io_mapping_section(config.get("io"), source_map=source_map)
    _require_spec(config.get("loss"), ("loss",), source_map=source_map, expected_prefix="loss.")
    _validate_optimizer_section(config.get("optimizer"), source_map=source_map)
    _validate_scheduler_section(config.get("scheduler"), source_map=source_map)

    _validate_list_of_specs(config.get("callbacks", []), ("callbacks",), source_map=source_map, expected_prefix="cb.")
    _validate_list_of_specs(config.get("loggers", []), ("loggers",), source_map=source_map, expected_prefix="log.")
    _validate_list_of_specs(config.get("metrics", []), ("metrics",), source_map=source_map, expected_prefix="metric.")

    return run


def validate_post_model(
    config: dict[str, Any],
    *,
    registry: Registry,
    source_map: SourceMap,
    model: torch.nn.Module,
) -> None:
    _assert_type_if_present(
        registry,
        spec=config["trainer"],
        spec_path=("trainer",),
        source_map=source_map,
        domain="trainer",
        prefix="trainer.",
    )
    _assert_type_if_present(
        registry,
        spec=config["data"],
        spec_path=("data",),
        source_map=source_map,
        domain="data",
        prefix="data.",
    )
    _assert_type_if_present(
        registry,
        spec=config["io"],
        spec_path=("io",),
        source_map=source_map,
        domain="io",
        prefix="io.",
    )
    _assert_type_if_present(
        registry,
        spec=config["loss"],
        spec_path=("loss",),
        source_map=source_map,
        domain="loss",
        prefix="loss.",
    )

    _assert_type_if_present(
        registry,
        spec=config["optimizer"],
        spec_path=("optimizer",),
        source_map=source_map,
        domain="optimizer factory",
        prefix="optim_factory.",
    )
    _assert_key_exists(
        registry,
        key=_require_str(config["optimizer"].get("type"), ("optimizer", "type"), source_map=source_map, required=True),
        config_path=("optimizer", "type"),
        source_map=source_map,
        domain="optimizer class",
        prefix="optim_cls.",
    )

    scheduler = config.get("scheduler")
    if scheduler is not None:
        assert isinstance(scheduler, dict)
        _assert_type_if_present(
            registry,
            spec=scheduler,
            spec_path=("scheduler",),
            source_map=source_map,
            domain="scheduler factory",
            prefix="sched_factory.",
        )
        _assert_key_exists(
            registry,
            key=_require_str(scheduler.get("type"), ("scheduler", "type"), source_map=source_map, required=True),
            config_path=("scheduler", "type"),
            source_map=source_map,
            domain="scheduler class",
            prefix="sched_cls.",
        )

    for i, cb in enumerate(config.get("callbacks") or []):
        _assert_type_if_present(
            registry,
            spec=cb,
            spec_path=("callbacks", i),
            source_map=source_map,
            domain="callback",
            prefix="cb.",
        )

    for i, lg in enumerate(config.get("loggers") or []):
        _assert_type_if_present(
            registry,
            spec=lg,
            spec_path=("loggers", i),
            source_map=source_map,
            domain="logger",
            prefix="log.",
        )

    for i, m in enumerate(config.get("metrics") or []):
        _assert_type_if_present(
            registry,
            spec=m,
            spec_path=("metrics", i),
            source_map=source_map,
            domain="metric",
            prefix="metric.",
        )

    _validate_mapping_io_against_model(config.get("io"), source_map=source_map, model=model)


def _validate_unique_imports(imports: Any, *, source_map: SourceMap) -> None:
    if not imports:
        return
    if not isinstance(imports, list) or not all(isinstance(x, str) for x in imports):
        raise_config_error("imports must be a list of strings", config_path=("imports",), source_map=source_map)
    seen: set[str] = set()
    for i, mod in enumerate(imports):
        if mod in seen:
            raise_config_error(f"Duplicate import {mod!r}", config_path=("imports", i), source_map=source_map)
        seen.add(mod)


def _validate_list_of_specs(items: Any, path: tuple[Any, ...], *, source_map: SourceMap, expected_prefix: str) -> None:
    if items is None:
        return
    if not isinstance(items, list):
        raise_config_error("must be a list", config_path=path, source_map=source_map)
    for i, item in enumerate(items):
        _require_spec(item, path + (i,), source_map=source_map, expected_prefix=expected_prefix)


def _require_mapping(
    obj: Any,
    path: tuple[Any, ...],
    *,
    source_map: SourceMap,
    required: bool = False,
) -> dict[str, Any]:
    if obj is None and not required:
        return {}
    if not isinstance(obj, dict):
        raise_config_error("must be a mapping", config_path=path, source_map=source_map)
    return obj


def _require_spec(obj: Any, path: tuple[Any, ...], *, source_map: SourceMap, expected_prefix: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise_config_error("must be a spec mapping", config_path=path, source_map=source_map)
    if "_type_" not in obj and "_target_" not in obj:
        raise_config_error("spec must contain _type_ or _target_", config_path=path, source_map=source_map)
    if "_type_" in obj:
        t = obj.get("_type_")
        if not isinstance(t, str) or not t:
            raise_config_error("_type_ must be a non-empty string", config_path=path + ("_type_",), source_map=source_map)
        if not t.startswith(expected_prefix):
            raise_config_error(
                f"_type_ must start with {expected_prefix!r}",
                config_path=path + ("_type_",),
                source_map=source_map,
            )
    return obj


def _require_type_key(obj: Any, path: tuple[Any, ...], *, source_map: SourceMap) -> str:
    if not isinstance(obj, dict) or "_type_" not in obj:
        raise_config_error("must be a _type_ spec", config_path=path, source_map=source_map)
    t = obj.get("_type_")
    if not isinstance(t, str) or not t:
        raise_config_error("_type_ must be a non-empty string", config_path=path + ("_type_",), source_map=source_map)
    return t


def _assert_type_if_present(
    registry: Registry,
    *,
    spec: Any,
    spec_path: tuple[Any, ...],
    source_map: SourceMap,
    domain: str,
    prefix: str,
) -> None:
    if not isinstance(spec, dict) or "_type_" not in spec:
        return
    key = _require_str(spec.get("_type_"), spec_path + ("_type_",), source_map=source_map, required=True)
    _assert_key_exists(
        registry,
        key=key,
        config_path=spec_path + ("_type_",),
        source_map=source_map,
        domain=domain,
        prefix=prefix,
    )


def _validate_optimizer_section(optimizer: Any, *, source_map: SourceMap) -> None:
    opt = _require_spec(optimizer, ("optimizer",), source_map=source_map, expected_prefix="optim_factory.")
    t = _require_str(opt.get("type"), ("optimizer", "type"), source_map=source_map, required=True)
    if not t.startswith("optim_cls."):
        raise_config_error("optimizer.type must start with 'optim_cls.'", config_path=("optimizer", "type"), source_map=source_map)

    args = opt.get("args", [])
    kwargs = opt.get("kwargs", {})
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    if not isinstance(args, list):
        raise_config_error("optimizer.args must be a list", config_path=("optimizer", "args"), source_map=source_map)
    if not isinstance(kwargs, dict):
        raise_config_error("optimizer.kwargs must be a mapping", config_path=("optimizer", "kwargs"), source_map=source_map)

    nested_path = _find_nested_spec(args)
    if nested_path is not None:
        raise_config_error(
            "Nested specs are not allowed inside optimizer.args (constructor refs must be pure data)",
            config_path=("optimizer", "args") + nested_path,
            source_map=source_map,
        )
    nested_path = _find_nested_spec(kwargs)
    if nested_path is not None:
        raise_config_error(
            "Nested specs are not allowed inside optimizer.kwargs (constructor refs must be pure data)",
            config_path=("optimizer", "kwargs") + nested_path,
            source_map=source_map,
        )


def _validate_scheduler_section(scheduler: Any, *, source_map: SourceMap) -> None:
    if scheduler is None:
        return
    sched = _require_spec(scheduler, ("scheduler",), source_map=source_map, expected_prefix="sched_factory.")

    t = _require_str(sched.get("type"), ("scheduler", "type"), source_map=source_map, required=True)
    if not t.startswith("sched_cls."):
        raise_config_error("scheduler.type must start with 'sched_cls.'", config_path=("scheduler", "type"), source_map=source_map)

    args = sched.get("args", [])
    kwargs = sched.get("kwargs", {})
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    if not isinstance(args, list):
        raise_config_error("scheduler.args must be a list", config_path=("scheduler", "args"), source_map=source_map)
    if not isinstance(kwargs, dict):
        raise_config_error("scheduler.kwargs must be a mapping", config_path=("scheduler", "kwargs"), source_map=source_map)

    interval = sched.get("interval", "epoch")
    if interval not in ("step", "epoch", "metric"):
        raise_config_error("scheduler.interval must be one of: step | epoch | metric", config_path=("scheduler", "interval"), source_map=source_map)
    frequency = sched.get("frequency", 1)
    if not isinstance(frequency, int) or frequency < 1:
        raise_config_error("scheduler.frequency must be an int >= 1", config_path=("scheduler", "frequency"), source_map=source_map)
    monitor = sched.get("monitor")
    if monitor is not None and not isinstance(monitor, str):
        raise_config_error("scheduler.monitor must be a string or null", config_path=("scheduler", "monitor"), source_map=source_map)
    if interval == "metric" and (monitor is None or not isinstance(monitor, str) or not monitor):
        raise_config_error("scheduler.monitor is required when interval=metric", config_path=("scheduler", "monitor"), source_map=source_map)

    nested_path = _find_nested_spec(args)
    if nested_path is not None:
        raise_config_error(
            "Nested specs are not allowed inside scheduler.args (constructor refs must be pure data)",
            config_path=("scheduler", "args") + nested_path,
            source_map=source_map,
        )
    nested_path = _find_nested_spec(kwargs)
    if nested_path is not None:
        raise_config_error(
            "Nested specs are not allowed inside scheduler.kwargs (constructor refs must be pure data)",
            config_path=("scheduler", "kwargs") + nested_path,
            source_map=source_map,
        )


def _find_nested_spec(obj: Any) -> tuple[Any, ...] | None:
    if isinstance(obj, dict):
        if "_type_" in obj or "_target_" in obj:
            return ()
        for k, v in obj.items():
            nested = _find_nested_spec(v)
            if nested is not None:
                return (k,) + nested
        return None
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            nested = _find_nested_spec(v)
            if nested is not None:
                return (i,) + nested
        return None
    return None


def _parse_run(run: dict[str, Any], *, source_map: SourceMap) -> RunConfig:
    name = _require_str(run.get("name"), ("run", "name"), source_map=source_map, required=True)
    out_dir = _require_str(run.get("out_dir"), ("run", "out_dir"), source_map=source_map, required=True)

    seed = run.get("seed", 0)
    if not isinstance(seed, int):
        raise_config_error("run.seed must be an int", config_path=("run", "seed"), source_map=source_map)

    deterministic = run.get("deterministic", True)
    if not isinstance(deterministic, bool):
        raise_config_error("run.deterministic must be a bool", config_path=("run", "deterministic"), source_map=source_map)

    device_raw = run.get("device", "cpu")
    if not isinstance(device_raw, str) or not device_raw:
        raise_config_error("run.device must be a non-empty string", config_path=("run", "device"), source_map=source_map)
    device = torch.device(device_raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise_config_error("CUDA requested but torch.cuda.is_available() is false", config_path=("run", "device"), source_map=source_map)

    precision = run.get("precision", "fp32")
    if precision not in ("fp32", "fp16-mixed", "bf16-mixed"):
        raise_config_error("run.precision must be one of: fp32 | fp16-mixed | bf16-mixed", config_path=("run", "precision"), source_map=source_map)
    if precision == "fp16-mixed" and device.type != "cuda":
        raise_config_error("fp16-mixed requires a CUDA device", config_path=("run", "precision"), source_map=source_map)

    resume_from = run.get("resume_from")
    if resume_from is not None and not isinstance(resume_from, str):
        raise_config_error("run.resume_from must be a string or null", config_path=("run", "resume_from"), source_map=source_map)

    return RunConfig(
        name=name,
        out_dir=out_dir,
        seed=seed,
        device=device,
        precision=precision,
        deterministic=deterministic,
        resume_from=resume_from,
    )


def _require_str(obj: Any, path: tuple[Any, ...], *, source_map: SourceMap, required: bool) -> str:
    if obj is None and not required:
        return ""
    if not isinstance(obj, str) or not obj:
        raise_config_error("must be a non-empty string", config_path=path, source_map=source_map)
    return obj


def _assert_key_exists(
    registry: Registry,
    *,
    key: str,
    config_path: tuple[Any, ...],
    source_map: SourceMap,
    domain: str,
    prefix: str,
) -> None:
    if not key.startswith(prefix):
        raise_config_error(f"must start with {prefix!r}", config_path=config_path, source_map=source_map)

    available = [k for k in registry.list_modules() if k.startswith(prefix)]
    if not available:
        raise_config_error(
            f"No registry entries available for {domain} (prefix {prefix!r}); did you forget imports?",
            config_path=config_path,
            source_map=source_map,
        )

    if key in available:
        return

    suggestions = get_close_matches(key, available, n=5)
    raise_config_error(f"Unknown {domain} {key!r}", config_path=config_path, source_map=source_map, suggestions=suggestions)


def _validate_mapping_io_against_model(io_spec: Any, *, source_map: SourceMap, model: torch.nn.Module) -> None:
    if not isinstance(io_spec, dict) or io_spec.get("_type_") != "io.mapping":
        return
    model_inputs = getattr(model, "inputs", None)
    if not isinstance(model_inputs, list) or not all(isinstance(x, str) for x in model_inputs):
        return

    model_kwargs = io_spec.get("model_kwargs")
    if not isinstance(model_kwargs, dict):
        raise_config_error("io.mapping.model_kwargs must be a mapping", config_path=("io", "model_kwargs"), source_map=source_map)
    keys = set(model_kwargs.keys())
    required = set(model_inputs)
    missing = sorted(required - keys)
    extra = sorted(keys - required)
    if missing:
        raise_config_error(f"Missing model input mappings: {missing}", config_path=("io", "model_kwargs"), source_map=source_map)
    if extra:
        raise_config_error(f"Unknown model input mappings: {extra}", config_path=("io", "model_kwargs"), source_map=source_map)


def _validate_io_mapping_section(io_spec: Any, *, source_map: SourceMap) -> None:
    if not isinstance(io_spec, dict) or io_spec.get("_type_") != "io.mapping":
        return

    model_kwargs = io_spec.get("model_kwargs")
    if not isinstance(model_kwargs, dict):
        raise_config_error("io.mapping.model_kwargs must be a mapping", config_path=("io", "model_kwargs"), source_map=source_map)
    for k, v in model_kwargs.items():
        if not isinstance(k, str) or not k:
            raise_config_error("io.mapping.model_kwargs keys must be non-empty strings", config_path=("io", "model_kwargs"), source_map=source_map)
        _validate_path(v, path=("io", "model_kwargs", k), source_map=source_map)

    targets = io_spec.get("targets")
    if not isinstance(targets, dict):
        raise_config_error("io.mapping.targets must be a mapping", config_path=("io", "targets"), source_map=source_map)
    for k, v in targets.items():
        if not isinstance(k, str) or not k:
            raise_config_error("io.mapping.targets keys must be non-empty strings", config_path=("io", "targets"), source_map=source_map)
        _validate_path(v, path=("io", "targets", k), source_map=source_map)


def _validate_path(obj: Any, *, path: tuple[Any, ...], source_map: SourceMap) -> None:
    if not isinstance(obj, list):
        raise_config_error("path must be a list of segments", config_path=path, source_map=source_map)
    for i, seg in enumerate(obj):
        if not isinstance(seg, (str, int)):
            raise_config_error(
                "path segments must be strings (dict keys) or ints (sequence indices)",
                config_path=path + (i,),
                source_map=source_map,
            )
