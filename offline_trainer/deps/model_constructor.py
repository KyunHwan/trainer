from __future__ import annotations

from typing import Any, Callable

try:
    from model_constructor import Registry, build_model, resolve_config
    from model_constructor.config.settings import Settings
    from model_constructor.config.source_map import SourceMap
    from model_constructor.errors import ConfigError, ModelConstructorError
    from model_constructor.instantiate.instantiate import instantiate_value
    from model_constructor.instantiate.signature import validate_kwargs
except ImportError:  # pragma: no cover
    from policy_constructor.model_constructor import Registry, build_model, resolve_config
    from policy_constructor.model_constructor.config.settings import Settings
    from policy_constructor.model_constructor.config.source_map import SourceMap
    from policy_constructor.model_constructor.errors import ConfigError, ModelConstructorError
    from policy_constructor.model_constructor.instantiate.instantiate import instantiate_value
    from policy_constructor.model_constructor.instantiate.signature import validate_kwargs

__all__ = [
    "Any",
    "Callable",
    "ConfigError",
    "ModelConstructorError",
    "Registry",
    "Settings",
    "SourceMap",
    "build_model",
    "instantiate_value",
    "resolve_config",
    "validate_kwargs",
]

