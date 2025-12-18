from __future__ import annotations

from pathlib import Path

import torch

from offline_trainer.experiment import run_experiment


def test_fit_and_resume(tmp_path: Path) -> None:
    out_dir = tmp_path / "runs" / "exp"
    exp1 = tmp_path / "exp1.yaml"
    exp1.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "model:",
                "  sequential:",
                "    inputs: [x]",
                "    layers:",
                "      - _type_: nn.LazyLinear",
                "        out_features: 16",
                f"run: {{name: exp, out_dir: {out_dir.as_posix()}, seed: 0, device: cpu, precision: fp32, deterministic: true}}",
                "data: {_type_: data.random_regression, batch_size: 4, steps_per_epoch: 2, val_steps_per_epoch: 1, x_shape: [32], y_shape: [16]}",
                "io: {_type_: io.mapping, model_kwargs: {x: [x]}, targets: {y: [y]}}",
                "loss: {_type_: loss.torch, fn: {_type_: nn.MSELoss}}",
                "optimizer: {_type_: optim_factory.torch, type: optim_cls.SGD, kwargs: {lr: 0.01}}",
                "trainer: {_type_: trainer.default, max_epochs: 1, log_every_n_steps: 100, val_every_n_epochs: 1}",
                "callbacks:",
                "  - _type_: cb.model_checkpoint",
                "    save_dir: ${run.out_dir}/checkpoints",
                "    save_last: true",
            ]
        ),
        encoding="utf-8",
    )

    run_experiment(exp1)
    ckpt_path = out_dir / "checkpoints" / "last.pt"
    raw1 = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    step1 = int(raw1["train_state"]["global_step"])

    exp2 = tmp_path / "exp2.yaml"
    exp2.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "model:",
                "  sequential:",
                "    inputs: [x]",
                "    layers:",
                "      - _type_: nn.LazyLinear",
                "        out_features: 16",
                f"run: {{name: exp, out_dir: {out_dir.as_posix()}, seed: 0, device: cpu, precision: fp32, deterministic: true, resume_from: {ckpt_path.as_posix()}}}",
                "data: {_type_: data.random_regression, batch_size: 4, steps_per_epoch: 2, val_steps_per_epoch: 1, x_shape: [32], y_shape: [16]}",
                "io: {_type_: io.mapping, model_kwargs: {x: [x]}, targets: {y: [y]}}",
                "loss: {_type_: loss.torch, fn: {_type_: nn.MSELoss}}",
                "optimizer: {_type_: optim_factory.torch, type: optim_cls.SGD, kwargs: {lr: 0.01}}",
                "trainer: {_type_: trainer.default, max_epochs: 2, log_every_n_steps: 100, val_every_n_epochs: 1}",
                "callbacks:",
                "  - _type_: cb.model_checkpoint",
                "    save_dir: ${run.out_dir}/checkpoints",
                "    save_last: true",
            ]
        ),
        encoding="utf-8",
    )

    run_experiment(exp2)
    raw2 = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    step2 = int(raw2["train_state"]["global_step"])
    assert step2 > step1
