# Data

Data modules follow a minimal protocol for setup and dataloader construction.

Files:
- `datamodule.py`: `DataModule` protocol and `RandomRegressionDataModule` built-in.
- `utils.py`: helpers for seeding dataloader workers.

Custom datamodules should return batches compatible with `train.model_input` selection.
