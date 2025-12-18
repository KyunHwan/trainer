from __future__ import annotations

from pathlib import Path

import pytest

from offline_trainer.deps.model_constructor import ConfigError
from offline_trainer.experiment import run_experiment


def test_duplicate_imports_error_has_yaml_path(tmp_path: Path) -> None:
    cfg = tmp_path / "exp.yaml"
    cfg.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "imports: [extensions.example_plugin, extensions.example_plugin]",
                "model:",
                "  sequential:",
                "    inputs: [x]",
                "    layers:",
                "      - _type_: nn.Identity",
                "run: {name: t, out_dir: runs/t, seed: 0, device: cpu, precision: fp32, deterministic: true}",
                "data: {_type_: data.random_regression, batch_size: 2, steps_per_epoch: 1, x_shape: [4], y_shape: [4]}",
                "io: {_type_: io.mapping, model_kwargs: {x: [x]}, targets: {y: [y]}}",
                "loss: {_type_: loss.torch, fn: {_type_: nn.MSELoss}}",
                "optimizer: {_type_: optim_factory.torch, type: optim_cls.SGD, kwargs: {lr: 0.01}}",
                "trainer: {_type_: trainer.default, max_epochs: 1, log_every_n_steps: 100, val_every_n_epochs: 1}",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError) as exc_info:
        run_experiment(cfg)

    assert exc_info.value.config_path == ("imports", 1)


def test_optimizer_nested_spec_is_rejected(tmp_path: Path) -> None:
    cfg = tmp_path / "exp.yaml"
    cfg.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "model:",
                "  sequential:",
                "    inputs: [x]",
                "    layers:",
                "      - _type_: nn.Identity",
                "run: {name: t, out_dir: runs/t, seed: 0, device: cpu, precision: fp32, deterministic: true}",
                "data: {_type_: data.random_regression, batch_size: 2, steps_per_epoch: 1, x_shape: [4], y_shape: [4]}",
                "io: {_type_: io.mapping, model_kwargs: {x: [x]}, targets: {y: [y]}}",
                "loss: {_type_: loss.torch, fn: {_type_: nn.MSELoss}}",
                "optimizer:",
                "  _type_: optim_factory.torch",
                "  type: optim_cls.SGD",
                "  kwargs:",
                "    lr: 0.01",
                "    something: {_type_: nn.Identity}",
                "trainer: {_type_: trainer.default, max_epochs: 1, log_every_n_steps: 100, val_every_n_epochs: 1}",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError) as exc_info:
        run_experiment(cfg)

    assert exc_info.value.config_path == ("optimizer", "kwargs", "something")

