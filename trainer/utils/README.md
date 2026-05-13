# Utils

This folder contains shared utilities used across the training stack. For project-wide context, see [docs/README.md](../../docs/README.md).

## Purpose

- Traverse and transform nested data structures (dicts, lists, tuples, dataclasses)
- Move tensors between devices and cast dtypes
- Seed all RNG sources for reproducible training
- Safely select values from structured objects
- Dynamically instantiate classes with filtered kwargs

## Layout

| File | Description |
|------|-------------|
| [`tree.py`](tree.py) | `tree_map(fn, tree)` — recursively applies `fn` to leaves in nested structures (dict, list, tuple, namedtuple, dataclass). `tree_flatten(tree)` collects all leaves into a flat list. `is_tensor_leaf` predicate for tensor-specific traversal |
| [`device.py`](device.py) | `select_device(requested)` — resolves `"auto"` to CUDA/CPU. `move_to_device(batch, device)` — moves all tensors in nested structure to device. `cast_dtype(batch, dtype)` — casts floating-point tensors to a given dtype. Both use `tree_map` internally |
| [`seed.py`](seed.py) | `set_global_seed(seed, deterministic)` — seeds Python `random`, NumPy, and PyTorch RNGs. Sets `PYTHONHASHSEED`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, and optionally enables deterministic mode. `seed_worker(worker_id)` — seeds dataloader workers based on `torch.initial_seed()` |
| [`selection.py`](selection.py) | `select(obj, key)` — retrieves a value by string key (dict), integer index (tuple/list), or attribute name. Returns `obj` unchanged when `key` is `None` |
| [`import_utils.py`](import_utils.py) | `import_module(path)` — thin wrapper around `importlib.import_module`. `instantiate(obj, params, **extra)` — calls a class/function with merged `params` and `extra` kwargs, filtering to only accepted parameters via signature inspection |

## Contracts

- `tree_map` accepts an optional `is_leaf` predicate to customize what counts as a leaf node. Default leaf detection treats anything that isn't a dict/list/tuple/namedtuple/dataclass as a leaf.
- `move_to_device` is a no-op for tensors already on the target device.
- `cast_dtype` only touches floating-point tensors. Integer tensors (`task_index`, `is_pad`) and non-tensor leaves pass through unchanged.
- `instantiate` filters kwargs by signature, so passing extra kwargs (e.g. `device=...` to a factory that doesn't declare it) is safe — they're silently dropped. The exception: if the target accepts `**kwargs`, all extras are forwarded.
- `set_global_seed(seed, deterministic=True)` additionally enables `torch.use_deterministic_algorithms(True)` and `cudnn.deterministic=True`, which can raise on ops without deterministic implementations.
- `seed_worker` is designed for use as the `worker_init_fn` parameter of `DataLoader`.

## How to extend

These utilities are foundational; extension is rarely needed. If you do need to:

- Custom leaf detection in `tree_map`: pass an `is_leaf` predicate.
- Custom device-move logic: write your own traversal using `tree_map`.

## Cross-links

- Seed protocol: [docs/02_distributed_training.md § The seed protocol](../../docs/02_distributed_training.md#the-seed-protocol)
- Per-batch pipeline: [docs/06_training_loop_walkthrough.md § Stage 10](../../docs/06_training_loop_walkthrough.md#stage-10-per-batch-pipeline)
- Hub: [docs/README.md](../../docs/README.md)

## Gotchas / invariants

- `move_to_device` is a no-op for already-on-device tensors (avoids unnecessary copies).
- `cast_dtype` only affects floating-point tensors — integer tensors are left unchanged. This is what allows the same `cast_dtype(data, torch.float32)` call in the loop to safely process batches that mix `float` image tensors and `long` indices.
- `set_global_seed` enables `torch.use_deterministic_algorithms(True)` only when `deterministic=True`, which may raise errors for operations without deterministic implementations.
- `seed_worker` is referenced by name in `_build_dataloader`; if you replace it, update the import.
