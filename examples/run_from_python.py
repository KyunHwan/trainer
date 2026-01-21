"""Example: run training from Python without a CLI framework."""
from __future__ import annotations

from trainer.api import train


if __name__ == "__main__":
    train("examples/configs/minimal.yaml")
