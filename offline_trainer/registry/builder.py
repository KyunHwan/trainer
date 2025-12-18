from __future__ import annotations

from offline_trainer.deps.model_constructor import Registry

from .builtins import register_offline_trainer_builtins

try:
    from model_constructor.blocks.register import register_blocks as register_model_blocks
    from model_constructor.registry.builtins import register_builtins as register_model_builtins
except ImportError:  # pragma: no cover
    from policy_constructor.model_constructor.blocks.register import register_blocks as register_model_blocks
    from policy_constructor.model_constructor.registry.builtins import register_builtins as register_model_builtins


def build_registry() -> Registry:
    registry = Registry()
    register_model_builtins(registry)
    register_model_blocks(registry)
    register_offline_trainer_builtins(registry)
    return registry

