from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure() -> None:
    root = Path(__file__).resolve().parents[1]
    policy_constructor = root / "policy_constructor"

    for p in (root, policy_constructor):
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))

