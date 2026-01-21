from collections import namedtuple
from dataclasses import dataclass

import torch

from trainer.utils.device import move_to_device


@dataclass
class Sample:
    tensor: torch.Tensor
    label: int


Pair = namedtuple("Pair", ["left", "right"])


def test_move_to_device_nested_structures() -> None:
    batch = {
        "x": torch.randn(2, 3),
        "meta": [
            Sample(torch.randn(1), 7),
            Pair(torch.randn(2), "keep"),
        ],
        "tag": "static",
    }

    out = move_to_device(batch, torch.device("cpu"))

    assert isinstance(out["meta"][0], Sample)
    assert out["meta"][0].label == 7
    assert out["meta"][1].right == "keep"
    assert out["x"].device.type == "cpu"
    assert out["meta"][0].tensor.device.type == "cpu"
    assert out["meta"][1].left.device.type == "cpu"
