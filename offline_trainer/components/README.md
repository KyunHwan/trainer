# Components

This package defines minimal interfaces for pluggable training components and provides built-ins.

Files:
- `optim.py`: optimizer factories, `build(params)` returns a torch optimizer.
- `sched.py`: scheduler factories, `build(optimizer)` returns a scheduler or None.
- `loss.py`: loss callables that consume a batch and model outputs.
- `metrics.py`: stateful metrics with `reset`, `update`, and `compute`.
- `callbacks.py`: training callbacks with state save/restore hooks.
- `loggers.py`: loggers that receive metric dicts and step counters.

Custom components should be registered via the registries in `offline_trainer/registry`.
