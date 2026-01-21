import os

import torch
import yaml

from trainer.api import train


def _write_config(tmp_path, config: dict) -> str:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(config))
    return str(cfg_path)


def test_checkpoint_resume(tmp_path) -> None:
    model_path = os.path.join(
        os.getcwd(), "policy_constructor", "configs", "examples", "sequential_mlp.yaml"
    )
    save_dir = tmp_path / "ckpt"

    base_config = {
        "model": {"config_path": model_path},
        "data": {
            "datamodule": {
                "type": "random_regression",
                "params": {"batch_size": 2, "dataset_size": 8, "input_dim": 8, "output_dim": 16},
            }
        },
    }

    config1 = {
        **base_config,
        "train": {
            "model_input": "x",
            "max_steps": 1,
            "max_epochs": 1,
            "checkpoint": {"save_dir": str(save_dir), "save_every_n_steps": 1},
        },
    }
    train(_write_config(tmp_path, config1))

    last_path = save_dir / "last.pt"
    assert last_path.exists()
    ckpt1 = torch.load(last_path, map_location="cpu")
    assert ckpt1["train_state"]["step"] == 1

    config2 = {
        **base_config,
        "train": {
            "model_input": "x",
            "max_steps": 2,
            "max_epochs": 1,
            "checkpoint": {
                "save_dir": str(save_dir),
                "save_every_n_steps": 1,
                "resume_from": str(last_path),
            },
        },
    }
    train(_write_config(tmp_path, config2))

    ckpt2 = torch.load(last_path, map_location="cpu")
    assert ckpt2["train_state"]["step"] == 2
