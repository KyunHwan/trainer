from __future__ import annotations

from pathlib import Path

from .experiment import run_experiment

__all__ = ["run_experiment", "run"]


def run(config_path: str | Path) -> None:
    run_experiment(config_path)

