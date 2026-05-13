# Distributed training (DDP)

## Contents

- [What DDP does, conceptually](#what-ddp-does-conceptually)
- [Launching with torchrun](#launching-with-torchrun)
- [How the trainer wires DDP](#how-the-trainer-wires-ddp)
- [The seed protocol](#the-seed-protocol)
- [DistributedSampler vs RandomSampler](#distributedsampler-vs-randomsampler)
- [SyncBatchNorm](#syncbatchnorm)
- [find_unused_parameters](#find_unused_parameters)
- [Rank-0-only side effects](#rank-0-only-side-effects)
- [Checkpoint unwrapping](#checkpoint-unwrapping)
- [Multi-node DDP](#multi-node-ddp)

## What DDP does, conceptually

Each rank holds an **independent replica** of the model on a single GPU. Every replica runs the same forward and backward pass on its own slice of the global batch. Before the optimizer step, gradients are **all-reduced** across replicas: each rank ends up with the average gradient over the global batch. Each rank then steps its own optimizer with that averaged gradient — because all ranks start from identical weights and apply the same averaged update, the replicas stay in sync without ever exchanging parameters.

Two consequences fall out of this:

1. The **global batch size** is `data.batch_size * world_size`. Per-rank batch size stays at `data.batch_size`. This is why you'll see `Global batch size = 60` printed by rank 0 at startup.
2. Any operation that breaks the "all ranks do the same work" symmetry — `if rank == 0: print(...)`, conditional model branches, control flow that depends on data — can desync replicas or deadlock collective ops. The codebase handles this by (a) gating side effects to rank 0 only and (b) carefully placing `dist.barrier()` calls.

## Launching with torchrun

```bash
torchrun --nproc_per_node=<NUM_GPUS> trainer/offline_trainer.py \
  --train_config <path/to/config.yaml>
```

`torchrun` sets a handful of environment variables that the trainer reads:

| Variable | Set by torchrun | Read by the trainer |
|---|---|---|
| `WORLD_SIZE` | Total ranks across all nodes | `train()` in [`offline_trainer.py`](../trainer/offline_trainer.py) decides DDP is on iff `WORLD_SIZE > 1` |
| `RANK` | Global rank index | Used via `dist.get_rank()` after init |
| `LOCAL_RANK` | Rank index *on this node* | Selects the CUDA device (`torch.cuda.set_device(local_rank)`) |
| `MASTER_ADDR`, `MASTER_PORT` | Coordination endpoint | Consumed by `init_process_group(init_method="env://")` |

The check at the top of `train()` is:

```python
world_size = int(os.environ.get("WORLD_SIZE", 1))
enable_dist_train = world_size > 1 and torch.cuda.is_available() and dist.is_available()
if enable_dist_train and torch.cuda.is_available():
    assert "LOCAL_RANK" in os.environ, "LOCAL_RANK missing; launch with torchrun."
```

Translation: if `WORLD_SIZE > 1` is in the environment but you ran `python ...` directly, the assertion fires immediately. Either use `torchrun` or `unset WORLD_SIZE` before running for single-GPU.

## How the trainer wires DDP

The relevant calls live in [`trainer/offline_trainer.py`](../trainer/offline_trainer.py):

| Step | Where | What |
|---|---|---|
| Process group init | `_dist_setup` | `dist.init_process_group(backend="nccl", init_method="env://")`. NCCL is the only backend used. |
| Device selection | `train()` | `torch.cuda.set_device(local_rank)` then `device = torch.device(f"cuda:{local_rank}")` |
| Model wrapping | `_build_models` | `models[k] = DDP(policy, find_unused_parameters=..., device_ids=[local_rank], output_device=local_rank)` |
| Cleanup | `_dist_cleanup` | `dist.destroy_process_group()` |

All four are no-ops when `enable_dist_train` is `False`.

## The seed protocol

The code seeds the RNG **twice**, before and after model construction:

```python
# Phase 1: same seed on all ranks
set_global_seed(seed=base_seed)
trainer = _build_trainer(...)        # weights initialize identically on every rank
_dist_barrier(...)

# Phase 2: per-rank seed offset
set_global_seed(seed=base_seed + rank)
_dist_barrier(...)
```

Both phases matter:

- **Phase 1 (identical seed)** guarantees that every rank computes the same weight initialization. Without this, DDP replicas would start from different weights and the first all-reduce would silently average them — leaving the model in a state none of the ranks ever "saw" during init.
- **Phase 2 (rank-offset seed)** gives each rank independent stochastic behavior at training time: different dropout masks, different image-augmentation choices, different shuffling of the data slice. Without this offset, every replica would dropout the same neurons every step, which removes the regularization variance you actually want from DDP.

The barrier between the phases is there to make sure no rank advances to Phase 2 before *all* ranks have finished model construction (otherwise an early rank could overwrite its RNG state mid-init on a peer).

`DataLoader` workers are seeded separately via `seed_worker` in [`trainer/utils/seed.py`](../trainer/utils/seed.py); it derives each worker's seed from `torch.initial_seed() + worker_id`, so workers within a rank are also distinct.

## DistributedSampler vs RandomSampler

`_build_dataloader` picks the sampler based on `enable_dist_train`:

```python
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=world_rank, drop_last=True) \
          if enable_dist_train else RandomSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler, ..., shuffle=False, drop_last=False)
```

Two things to notice:

1. **`shuffle=False` on the DataLoader** is mandatory when `sampler` is non-`None`. Shuffling is the sampler's job; setting both raises a PyTorch error. `DistributedSampler` shuffles internally based on `set_epoch(epoch)`, which is called once at the top of every epoch (`if enable_dist_train: sampler.set_epoch(epoch)`).
2. **`drop_last=True` on the sampler** ensures every rank gets the same number of batches per epoch. Without this, ranks could iterate different numbers of times and deadlock on the next collective op.

If you swap in your own dataset factory, you don't need to do anything sampler-related — `_build_dataloader` constructs the sampler from your dataset.

## SyncBatchNorm

If any `BatchNorm*d` layer is detected in a model and DDP is enabled, the model is converted in place:

```python
if torch.cuda.is_available() and enable_dist_train and any(
        isinstance(m, nn.modules.batchnorm._BatchNorm) for m in policy.modules()):
    policy = nn.SyncBatchNorm.convert_sync_batchnorm(policy)
```

This is done **before** `.to(device)` and before the DDP wrap. With `SyncBatchNorm`, running mean and variance are computed over the *global* batch each step (via all-reduce), not over each rank's local slice. That matters whenever per-rank batches are small — local BN stats are noisy and biased toward the rank's data shard.

You don't opt in or out; the framework decides based on whether BN layers are present.

## find_unused_parameters

```yaml
model:
  find_unused_parameters: true
```

This is required (no default). It is passed directly to `DDP(..., find_unused_parameters=...)`.

**What it costs**: when `True`, DDP traces every backward pass to detect parameters that did not receive a gradient and excludes them from the all-reduce. This trace is non-trivially expensive — a forward+backward step is noticeably slower.

**When you need it**: any architecture where parameter usage depends on the input. The clearest case in this codebase is mixture-of-experts: the gating network routes each sample to a subset of experts, so on any given step, some experts produce zero gradient. Without `find_unused_parameters=True`, DDP would hang at the synchronization point waiting for gradients that never arrive.

If your config has no MoE and no conditional branches, set it to `false` for a noticeable speedup.

## Rank-0-only side effects

The trainer carefully gates these to rank 0:

| Side effect | Where | Why |
|---|---|---|
| `wandb.init / log / finish` | `train()` and `_record` | Only one process should write to a single wandb run |
| `dataset_stats.pkl` write | `_build_dataloader` | Single file; concurrent writes would corrupt it |
| Checkpoint writes (`_save_checkpoints`) | `train()` | Same file path on a shared filesystem; multiple writers would race |
| `print("Parameters of ...")`, `print("Global batch size ...")` | `train()` and `_build_models` | Console clarity |

Every other line in the loop — forward, backward, optimizer step, dataloader iteration, barriers — runs on **every** rank. Crucially, `iterations += 1` is **not** gated:

```python
if rank == 0:
    _record(loss_dict, iterations, num_iter_per_epoch)
iterations += 1  # has to be updated for all GPUs
```

If you add a `continue` or `return` inside a rank-0 branch in this loop, you will desync the iteration counter across ranks and the next save-every check will fire on different epochs on different ranks. Don't.

## Checkpoint unwrapping

`_save_checkpoints` in [`trainer/offline_trainer.py`](../trainer/offline_trainer.py) saves the **unwrapped** module:

```python
state_to_save = models[key].module.state_dict() if isinstance(models[key], DDP) else models[key].state_dict()
```

This keeps the saved `.pt` files compatible with non-DDP loads: you can resume on one GPU after training on eight without any unwrap dance. Symmetrically, `_build_models` calls `policy.load_state_dict(...)` **before** the DDP wrap, so the loaded weights are also untouched-by-DDP.

## Multi-node DDP

The trainer reads `WORLD_SIZE`, `LOCAL_RANK`, `RANK`, `MASTER_ADDR`, and `MASTER_PORT` from the environment, all of which `torchrun` populates correctly in multi-node setups (`--nnodes`, `--node_rank`, `--rdzv_endpoint`). Single-node DDP is the configuration that has been used and exercised against this codebase; the multi-node path uses the same code with no extra branches. There is nothing trainer-specific blocking multi-node, but there is also no test or operational doc that confirms it works end-to-end.

> TODO (maintainer): confirm multi-node DDP has been run with this entrypoint, and document any cluster-specific NCCL env (`NCCL_IB_DISABLE`, `NCCL_SOCKET_IFNAME`, etc.) that's required at your sites.
