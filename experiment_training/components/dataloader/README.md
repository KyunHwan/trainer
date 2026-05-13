# dataloader

This folder contains dataset factory implementations for loading training data. For project-wide context, see [docs/README.md](../../../docs/README.md).

## Purpose

- Load and prepare datasets for training (LeRobot HuggingFace datasets, local HDF5 episodic data)
- Compute and return normalization statistics alongside the dataset
- Apply image augmentations (color jitter, Gaussian blur, rotation)

## Layout

| File | Registry key | Description |
|------|-------------|-------------|
| [`lerobot_data.py`](lerobot_data.py) | `lerobot_dataset_factory` | Loads datasets from the [LeRobot](https://github.com/huggingface/lerobot) library. Supports configurable action horizons, observation history windows, and `delta_timestamps`. Applies `ColorJitter` + `GaussianBlur` augmentations. Returns dataset + normalization stats from `dataset.meta.stats` |
| [`resfit_lerobot_data.py`](resfit_lerobot_data.py) | `resfit_lerobot_dataset_factory` | LeRobot variant tuned for the ResFit residual-policy setup. Adds a `reward_horizon` parameter and uses different observation sampling strides |
| [`episodic_data.py`](episodic_data.py) | `episodic_dataset_factory` | Loads episodic demonstration data from local HDF5 files. Supports multiple camera streams, image compression/decompression, configurable action chunks, temporal delays, and image downsampling. Computes normalization stats from the data |
| [`utils/`](utils/) | — | Shared data utilities (`config_loader.ConfigLoader`, `compute_norm_stats`, `find_all_hdf5`, `validate_hdf5_file`, `get_episode_len`) |

## Contracts

Each factory implements the `DatasetFactory` protocol:

```python
@DATASET_BUILDER_REGISTRY.register("key")
class MyDatasetFactory:
    def build(self, opt_params: dict | None, params: dict) -> dict[str, Any]:
        return {"dataset": dataset, "norm_stats": stats}
```

- `opt_params` is filled in by the trainer: `{'local_rank': int, 'dist_enabled': bool, 'save_dir': str}`.
- `params` is whatever the YAML `data.datamodule.params` block contains.
- The return must include a `"dataset"` key (a `torch.utils.data.Dataset`). A `"norm_stats"` key is optional but recommended — its value is pickled to `{save_dir}/dataset_stats.pkl` by rank 0 and made available to the trainer as `stats=` in `train_step`.

## How to extend

Implement a new `DatasetFactory` to support other data formats (RoboSet, RLDS, custom). The recipe is in [docs/07_extending.md § Recipe: a new dataset factory](../../../docs/07_extending.md#recipe-a-new-dataset-factory).

Normalization-stats format must follow `{key: {"mean": list-or-tensor, "std": list-or-tensor}}` for compatibility with the `tree_map(map_list_to_torch, ...)` conversion at the start of the training loop.

## Cross-links

- Recipe: [docs/07_extending.md § Recipe: a new dataset factory](../../../docs/07_extending.md#recipe-a-new-dataset-factory)
- Data dict shape: [docs/04_concepts.md § The data dict](../../../docs/04_concepts.md#the-data-dict)
- Normalization: [docs/04_concepts.md § The stats dict and where normalization lives](../../../docs/04_concepts.md#the-stats-dict-and-where-normalization-lives)
- Hub: [docs/README.md](../../../docs/README.md)

## Gotchas / invariants

- Normalization stats are returned as plain Python lists or tensors from the dataset. The training loop converts them to tensors via `tree_map(map_list_to_torch, stats)` before use. The framework does **not** apply the normalization — trainers do that themselves inside `train_step`.
- When `local_files_only: true`, `lerobot_data.py` sets `HF_HUB_OFFLINE=1` and expects the dataset to be cached at `root`.
- The episodic data loader computes normalization stats from up to 100k samples for efficiency (see `compute_norm_stats` in [`utils/utils.py`](utils/utils.py)).
- `delta_timestamps` in LeRobot are computed from `HZ`, `action_horizon`, and observation history params. They define the temporal windows for each data key.
