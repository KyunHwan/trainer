from offline_trainer.registry import DATAMODULE_REGISTRY, TRAINER_REGISTRY, register_builtins
from offline_trainer.registry.plugins import load_plugins


def test_plugin_import_registers_keys() -> None:
    register_builtins()
    load_plugins([
        "examples.extensions.custom_trainer",
        "examples.extensions.custom_data",
    ])
    assert TRAINER_REGISTRY.has("custom_trainer_v1")
    assert DATAMODULE_REGISTRY.has("custom_data_v1")
