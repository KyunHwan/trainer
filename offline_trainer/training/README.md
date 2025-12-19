# Training

Training logic is centered on a simple loop and a default trainer wrapper.

Modules:
- `trainer.py`: `Trainer` protocol and `DefaultTrainer` implementation.
- `loop.py`: core training loop with gradient accumulation, AMP, EMA, and logging.
- `checkpointing.py`: save and resume checkpoints.
- `state.py`: `TrainState` used by callbacks and checkpointing.
- `amp.py`: GradScaler helper.

The default loop expects a `models` dict with a "main" model and a datamodule that yields batches aligned with `train.model_input`.
