# Training loop walkthrough

A stage-by-stage tour of `train()` in [`trainer/offline_trainer.py`](../trainer/offline_trainer.py). Read [04_concepts.md](04_concepts.md) first if you haven't.

## Contents

- [Stage 1: Load and validate the config](#stage-1-load-and-validate-the-config)
- [Stage 2: Load plugins](#stage-2-load-plugins)
- [Stage 3: Resolve distributed state and device](#stage-3-resolve-distributed-state-and-device)
- [Stage 4: Initialize the process group](#stage-4-initialize-the-process-group)
- [Stage 5: wandb init (rank 0)](#stage-5-wandb-init-rank-0)
- [Stage 6: Seed protocol](#stage-6-seed-protocol)
- [Stage 7: Build the trainer](#stage-7-build-the-trainer)
- [Stage 8: Build the dataloader](#stage-8-build-the-dataloader)
- [Stage 9: The epoch loop](#stage-9-the-epoch-loop)
- [Stage 10: Per-batch pipeline](#stage-10-per-batch-pipeline)
- [Stage 11: Rank-0 logging and checkpoint save](#stage-11-rank-0-logging-and-checkpoint-save)
- [Stage 12: Cleanup](#stage-12-cleanup)

## Stage 1: Load and validate the config

```python
raw = load_config(config_path)
config: ExperimentConfig = validate_config(raw)
```

- `load_config` (see [`trainer/config/loader.py`](../trainer/config/loader.py)) does recursive defaults composition and deep merge. Returns a plain `dict`.
- `validate_config` (see [`trainer/config/schemas.py`](../trainer/config/schemas.py)) runs Pydantic. Raises `ConfigError` on failure with a formatted list of `ConfigValidationIssue`s.

After this stage, `config` is a typed `ExperimentConfig` instance. See [05_configuration.md](05_configuration.md) for every field.

## Stage 2: Load plugins

```python
load_plugins(config.plugins)
```

`load_plugins` (in [`trainer/registry/plugins.py`](../trainer/registry/plugins.py)) imports each module path via `importlib.import_module`. Each import triggers any module-scope `@TRAINER_REGISTRY.register(...)`, `@LOSS_BUILDER_REGISTRY.register(...)`, etc., populating the global registries. After this stage, all `type:` keys in the YAML must resolve via `<registry>.get(<key>)` — otherwise the next stage fails.

## Stage 3: Resolve distributed state and device

```python
world_size = int(os.environ.get("WORLD_SIZE", 1))
enable_dist_train = world_size > 1 and torch.cuda.is_available() and dist.is_available()

if enable_dist_train and torch.cuda.is_available():
    assert "LOCAL_RANK" in os.environ, "LOCAL_RANK missing; launch with torchrun."
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
else:
    local_rank = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

DDP is enabled iff three conditions hold: `WORLD_SIZE > 1`, CUDA available, `torch.distributed` available. Without all three the trainer runs as single-process / single-device (CUDA if available, CPU fallback otherwise).

The `LOCAL_RANK` assertion is the canonical "you launched with `python` instead of `torchrun`" failure — see [10_troubleshooting.md](10_troubleshooting.md).

## Stage 4: Initialize the process group

```python
_dist_setup(enable_dist_train, device)
# -> dist.init_process_group(backend="nccl", init_method="env://")
```

NCCL is the only backend. `init_method="env://"` reads `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK` from the environment — all populated by `torchrun`. After this returns, `dist.get_rank()` and `dist.get_world_size()` are valid; `train()` reads them next:

```python
rank = dist.get_rank() if enable_dist_train else 0
world_size = dist.get_world_size() if enable_dist_train else 1
```

When DDP is disabled, both are forced to `0` / `1` so the rest of the code can use them uniformly.

## Stage 5: wandb init (rank 0)

```python
if rank == 0:
    project_name = config.data.datamodule.params["task_name"]
    wandb.init(
        project=project_name,
        config=_params_dict(config),
        name=config.train.project_name,
    )
```

The wandb project is **`config.data.datamodule.params["task_name"]`** (e.g. `picknplace`). The run name is **`config.train.project_name`**. Only rank 0 calls `wandb.init`; logging in later stages is also gated to rank 0.

If `task_name` isn't present in your datamodule params, this raises `KeyError`. Today every dataset factory's YAML must include it.

## Stage 6: Seed protocol

```python
base_seed = getattr(config.train, "seed", 0)
set_global_seed(seed=base_seed)
if rank == 0: print(f"Global batch size = {config.data.batch_size * world_size}")
trainer = _build_trainer(...)                       # weights init identical on every rank
_dist_barrier(enable_dist_train, local_rank)

set_global_seed(seed=base_seed + rank)              # per-rank offset for runtime stochasticity
_dist_barrier(enable_dist_train, local_rank)
dataloader, sampler, stats = _build_dataloader(...)
_dist_barrier(enable_dist_train, local_rank)
```

See [02_distributed_training.md § The seed protocol](02_distributed_training.md#the-seed-protocol) for the rationale on why both phases (sync then offset) are needed, with a barrier between them. `set_global_seed` is in [`trainer/utils/seed.py`](../trainer/utils/seed.py); it seeds Python `random`, NumPy, `torch.manual_seed`, `torch.cuda.manual_seed_all`, and sets `PYTHONHASHSEED`.

## Stage 7: Build the trainer

```python
trainer = _build_trainer(world_size, rank, local_rank, enable_dist_train, config, device)
```

[`_build_trainer`](../trainer/offline_trainer.py) is a thin orchestrator:

1. `models = _build_models(...)` — calls `PolicyConstructorModelFactory.build` on `config.model.component_config_paths`, then per component: optionally load from `train.load_dir`, optionally apply `init_weights`, optionally freeze, optionally convert to `SyncBatchNorm`, move to device, optionally wrap in DDP. Returns `nn.ModuleDict`. See [04_concepts.md § The nn.ModuleDict convention](04_concepts.md#the-nnmoduledict-convention).
2. `optimizers = _build_optimizers(config, models, device)` — for each entry in `config.model.component_optims`, look up the factory in `OPTIMIZER_BUILDER_REGISTRY`, build with the model's parameters. Skips components with no trainable parameters (frozen). Loads `<load_dir>/<name>_opt.pt` if present.
3. `loss_fn = _build_loss(config, device)` — registry lookup on `config.train.loss.type`, instantiate, call `.build()`, move to device.
4. `trainer = instantiate(trainer_cls, params, models=models, optimizers=optimizers, loss=loss_fn, device=device)` — the registered trainer class is instantiated with filtered kwargs.
5. `isinstance(trainer, Trainer)` is checked; mismatch raises `TypeError("Constructed object does not match Trainer interface")`.

## Stage 8: Build the dataloader

```python
dataloader, sampler, stats = _build_dataloader(
    config=config, world_rank=rank, local_rank=local_rank,
    world_size=world_size, enable_dist_train=enable_dist_train,
)
```

[`_build_dataloader`](../trainer/offline_trainer.py):

1. Looks up the dataset factory in `DATASET_BUILDER_REGISTRY[config.data.datamodule.type]`.
2. Instantiates it (via `instantiate`) and calls `.build(opt_params, params)` where `opt_params = {'local_rank': local_rank, 'dist_enabled': enable_dist_train, 'save_dir': config.train.save_dir}`.
3. If the returned dict has a `'norm_stats'` key, rank 0 pickles it to `{save_dir}/dataset_stats.pkl`.
4. Constructs a `DistributedSampler` (if DDP) or `RandomSampler` (otherwise) over the dataset.
5. Wraps in a `DataLoader` with `worker_init_fn=seed_worker`, `shuffle=False`, `drop_last=False`. Sampler `drop_last=True` only when DDP.

After this stage, `stats` is a Python dict of CPU lists/tensors. The loop will convert it to GPU tensors once, outside the inner per-batch loop.

## Stage 9: The epoch loop

```python
num_iter_per_epoch = float(len(dataloader))
stats_cpu = tree_map(map_list_to_torch, stats)
iterations = 0

for epoch in range(config.train.epoch):
    if enable_dist_train:
        sampler.set_epoch(epoch)
    for _, data in enumerate(tqdm(dataloader, disable=(rank != 0))):
        ...
```

- `num_iter_per_epoch` is the number of steps per epoch, used to derive the `'epoch'` value logged to wandb.
- `tree_map(map_list_to_torch, stats)` converts the nested dict of Python lists to a nested dict of CPU tensors. The trainer then casts/moves them to GPU inside the inner loop (see Stage 10).
- `sampler.set_epoch(epoch)` reshuffles the `DistributedSampler` deterministically based on the epoch number. Skipping this call would make every epoch see data in the same order on each rank.
- `tqdm(..., disable=(rank != 0))` keeps the progress bar on rank 0 only.

## Stage 10: Per-batch pipeline

```python
stats_gpu = cast_dtype(stats_cpu, torch.float32)
stats_gpu = move_to_device(stats_gpu, device)

data = cast_dtype(data, torch.float32)
data = move_to_device(data, device)

with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    loss_dict = trainer.train_step(data=data, stats=stats_gpu)
if rank == 0:
    _record(loss_dict, iterations, num_iter_per_epoch)
iterations += 1   # has to be updated for all GPUs
```

Step-by-step:

1. **`cast_dtype` + `move_to_device`** ([`trainer/utils/device.py`](../trainer/utils/device.py)) walk the nested dict and cast/move every tensor leaf. `cast_dtype` only touches floating-point tensors (integer tensors pass through unchanged — `task_index`, `is_pad`, etc.). `move_to_device` is a no-op for tensors already on `device`.
2. **`stats_gpu` is recomputed inside the loop**. This is wasteful — the stats don't change — but it's cheap and harmless. The online trainer hoists this outside the loop.
3. **`torch.autocast(device_type="cuda", dtype=torch.bfloat16)`** runs the trainer's forward and backward under bfloat16 mixed precision. bfloat16 has the same exponent range as float32, so it avoids the gradient-scaling complexity of float16 — no `GradScaler` is needed.
4. **`trainer.train_step(data=data, stats=stats_gpu)`** is called. The trainer is expected to zero grads, forward, backward, clip, step, and return a dict of metrics. See [04_concepts.md § The train_step contract](04_concepts.md#the-train_step-contract).
5. **`_record(loss_dict, iterations, num_iter_per_epoch)`** on rank 0 only — detaches tensor values, appends `'epoch'`, calls `wandb.log(step=iterations)`. See [09_experiment_tracking.md](09_experiment_tracking.md).
6. **`iterations += 1` is on every rank** — see the warning in [02_distributed_training.md § Rank-0-only side effects](02_distributed_training.md#rank-0-only-side-effects).

## Stage 11: Rank-0 logging and checkpoint save

After the inner loop, still inside the outer epoch loop:

```python
if rank == 0:
    print(f"Epoch {epoch} complete")
    if (epoch + 1) % config.train.save_every == 0:
        _save_checkpoints(models=trainer.models,
                          optimizers=trainer.optimizers,
                          save_dir=config.train.save_dir,
                          epoch=epoch + 1)
gc.collect()
torch.cuda.empty_cache()
_dist_barrier(enable_dist_train, local_rank)
```

- The save trigger is **(epoch + 1) % save_every == 0**, so epochs are 1-indexed when written to disk (`epoch_3/`, `epoch_6/`, ...).
- `_save_checkpoints` (see [08_checkpoints_and_resume.md](08_checkpoints_and_resume.md)) creates `epoch_<N>/`, unwraps DDP, saves `<component>.pt` and `<component>_opt.pt` files.
- `gc.collect()` + `torch.cuda.empty_cache()` runs on every rank to reduce peak memory between epochs.
- The trailing barrier guarantees no rank starts the next epoch before all have finished saving / cleaning up.

## Stage 12: Cleanup

```python
finally:
    if rank == 0:
        print("program terminating...")
        wandb.finish()
    _dist_cleanup(enable_dist_train)
```

The `finally` runs whether training completed normally, was interrupted, or raised. `wandb.finish()` flushes any buffered metrics. `_dist_cleanup` calls `dist.destroy_process_group()` to release NCCL handles.

If you hit an exception **inside** the epoch loop, this block still runs — but anything between the exception and the `finally` is skipped, so you may end up with a partially-written `epoch_<N>/` directory. The save is atomic at the level of individual `.pt` files (one `torch.save` call each), but not across components.
