import pytest

from trainer.config.errors import ConfigError
from trainer.config.schemas import validate_config


def test_config_errors_include_yaml_path() -> None:
    raw = {
        "model": {"config_path": "policy_constructor/configs/examples/sequential_mlp.yaml"},
        "train": {"optimizer": {"type": "adamw", "params": {"lr": -1.0}}},
    }
    with pytest.raises(ConfigError) as excinfo:
        validate_config(raw)
    msg = str(excinfo.value)
    assert "train.optimizer.params.lr" in msg
    assert "lr must be a float > 0" in msg
