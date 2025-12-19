import os

import yaml

from offline_trainer.api import train


def test_minimal_train_smoke(tmp_path) -> None:
    model_path = os.path.join(
        os.getcwd(), "policy_constructor", "configs", "examples", "sequential_mlp.yaml"
    )
    config = {
        "model": {"config_path": model_path},
        "data": {
            "datamodule": {
                "type": "random_regression",
                "params": {"batch_size": 2, "dataset_size": 8, "input_dim": 8, "output_dim": 16},
            }
        },
        "train": {
            "model_input": "x",
            "max_steps": 2,
            "max_epochs": 1,
            "checkpoint": {"save_dir": str(tmp_path), "save_every_n_steps": 1},
        },
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(config))

    train(str(cfg_path))
