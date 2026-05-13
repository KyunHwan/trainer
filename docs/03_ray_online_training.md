# Ray online/offline hybrid training

## Contents

- [What this mode is for](#what-this-mode-is-for)
- [Required Ray actors](#required-ray-actors)
- [Launching: it is not a script](#launching-it-is-not-a-script)
- [How it differs from the offline trainer](#how-it-differs-from-the-offline-trainer)
- [The replay buffer warmup](#the-replay-buffer-warmup)
- [Combining offline and online batches](#combining-offline-and-online-batches)
- [Save cadence and weight broadcasting](#save-cadence-and-weight-broadcasting)
- [Known oddities](#known-oddities)

## What this mode is for

The online trainer in [`trainer/online_trainer.py`](../trainer/online_trainer.py) targets **continuous training where the dataset is being collected and updated live**. Each training step mixes a batch from a fixed offline dataset with a batch sampled from a live Ray-backed replay buffer, then periodically pushes updated weights back out to inference workers via a second Ray actor.

You almost certainly do **not** want this mode unless you have:

- A Ray cluster (single node is fine for development) with `ray` installed,
- A populated `replay_buffer` named actor that supports the contract below,
- A `policy_state_manager` named actor (typically driven by your inference fleet) that wants weight updates.

For everything else, use the offline trainer ([01_getting_started.md](01_getting_started.md), [02_distributed_training.md](02_distributed_training.md)).

## Required Ray actors

The trainer expects two **named Ray actors** to already exist in the cluster when `train_func` starts. It fetches them by name and crashes immediately if they aren't there:

```python
replay_buffer = ray.get_actor("replay_buffer")
policy_state_manager = ray.get_actor("policy_state_manager")
```

### `replay_buffer` contract

The trainer calls exactly two methods on it:

- **`replay_buffer.size.remote() -> int`** — returns the current number of samples in the buffer. Used during the warmup loop to wait until there's enough data.
- **`replay_buffer.sample.remote(batch_size: int) -> dict[str, Tensor]`** — returns a batch as a dict of tensors. The dict must include every key the trainer expects to find in the **offline** dataset (`action`, `observation.state`, `observation.current`, `observation.images.cam_head`, etc. — see [04_concepts.md](04_concepts.md) for the canonical key list). Extra keys are allowed; only the intersection of offline and online keys is concatenated.

### `policy_state_manager` contract

Called only at save time:

- **`policy_state_manager.update_state.remote(weights_ref: ObjectRef) -> None`** — receives a Ray `ObjectRef` produced by `ray.put(...)` whose value is a `dict[component_name -> dict[param_name -> CPU Tensor]]`. The actor pulls the dict out of the object store and distributes it to inference workers (implementation lives outside this repo).

> TODO (maintainer): point readers at the canonical `replay_buffer` and `policy_state_manager` implementations or repos. Those actors live outside this trainer repo; the trainer just assumes the contracts above.

## Launching: it is not a script

Unlike `offline_trainer.py`, `online_trainer.py` has no `if __name__ == "__main__":` block. `train_func` is meant to be invoked by `ray.train.TorchTrainer` (or equivalent Ray-Train scaffolding) on each worker, passing the config path string. A typical launcher elsewhere will look like:

```python
import ray
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
from trainer.online_trainer import train_func

ray.init(...)
# ... assume replay_buffer and policy_state_manager actors are already up ...

trainer = TorchTrainer(
    train_loop_per_worker=lambda: train_func("/path/to/config.yaml"),
    scaling_config=ScalingConfig(num_workers=4, use_gpu=True),
)
trainer.fit()
```

> TODO (maintainer): include or link the actual launcher used in your deployments. The trainer file alone is not enough to start a job.

## How it differs from the offline trainer

The conceptual frame and the per-component construction (`_build_loss`, `_build_models`, `_build_optimizers`, `_build_trainer`) are nearly identical. The differences are concentrated in three places.

| Concern | Offline | Online |
|---|---|---|
| Distributed init | `dist.init_process_group(backend="nccl", init_method="env://")` in `_dist_setup` | Ray Train handles process-group init; the trainer just reads context from `ray.train.get_context()` |
| Model wrapping | `DDP(policy, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=...)` | `ray.train.torch.prepare_model(model=policy, parallel_strategy_kwargs={"find_unused_parameters": ...})` |
| DataLoader | Constructed with `DistributedSampler` (when `world_size > 1`) | Constructed plainly and then passed through `ray.train.torch.prepare_data_loader(...)`, which inserts the sampler under the hood |
| Loop shape | `for epoch in range(config.train.epoch): for batch in dataloader: ...` | `while True:` with manual offline-iterator advancement and inline `epoch` counter |
| Save cadence | Every `save_every` **epochs** | Every `save_every * 25` **iterations** (see [Save cadence](#save-cadence-and-weight-broadcasting) below) |
| Seed protocol | Same seed → barrier → rank-offset seed → barrier | **Same** two-phase protocol; verified at `train_func` lines that mirror the offline path |
| Cleanup | `_dist_cleanup` calls `destroy_process_group` | No `destroy_process_group` call — Ray Train owns the process group |

Everything else — Pydantic config validation, plugin loading, registry-driven instantiation, the `_build_*` helpers, normalization-stats persistence, wandb conventions — is shared.

## The replay buffer warmup

Before entering the training loop, every rank polls the buffer until there's enough data:

```python
replay_buffer_size = 0
while replay_buffer_size < config.data.batch_size * 2 * world_size:
    replay_buffer_size = ray.get(replay_buffer.size.remote())
    time.sleep(0.5)
```

The threshold is `2 × batch_size × world_size`. Once it's reached, all ranks pass the barrier and start training. If the buffer never reaches this size (e.g., your collector isn't running), the trainer will print `replay buffer size: <n>` forever on rank 0 — see the corresponding entry in [10_troubleshooting.md](10_troubleshooting.md).

## Combining offline and online batches

Every training step pulls *both* sources and concatenates them along the batch dimension:

1. **Offline** comes from `next(offline_iter)` (a normal PyTorch dataloader iterator). When exhausted, the iterator is restarted with `dataloader.sampler.set_epoch(epoch)` and `epoch += 1`.
2. **Online** comes from `ray.get(replay_buffer.sample.remote(batch_size=...))`.

Then the trainer takes the intersection of keys (`shared_keys = offline_data.keys() & online_data.keys()`) and concatenates:

- **For camera keys** (key contains `"cam"`): the online tensor is resized to the offline tensor's spatial shape via `torchvision.transforms.Resize(target_size, antialias=True)`, and both sides are promoted to `float32` if they came in as `uint8`. The resize correctly handles both `(B, C, H, W)` and `(B, T, C, H, W)` by reshaping to 4D, resizing, and restoring the leading dimensions.
- **For all other keys**: a plain `torch.cat([offline_data[key], online_data[key]], dim=0)`.

If `base_policy_action` appears in `online_data` (a `resfit`-style residual signal), the trainer also adds `offline_data['base_policy_action'] = offline_data['action'].detach().clone()` so the concatenation works.

The combined dict, plus an `'iter'` key with the current iteration number, becomes the `data` argument to `trainer.train_step(data=data, stats=stats_gpu)`. Normalization stats are computed once on CPU from the offline dataset, moved to GPU once before the loop, and reused every step.

## Save cadence and weight broadcasting

```python
if (iterations + 1) % (config.train.save_every * 25) == 0:
    _save_checkpoints(models=trainer.models, optimizers=trainer.optimizers,
                      save_dir=config.train.save_dir, epoch=epoch)

    policy_components_weights = {}
    for model_name in trainer.models.keys():
        if not config.model.component_build_args[model_name]['freeze'] \
                and config.model.component_build_args[model_name]['online_update']:
            raw_model = unwrap_model(trainer.models[model_name])
            policy_components_weights[model_name] = {k: v.cpu() for k, v in raw_model.state_dict().items()}

    weights_ref = ray.put(policy_components_weights)
    policy_state_manager.update_state.remote(weights_ref)
```

Two things to remember:

- **Cadence is `save_every × 25` iterations**, not epochs. If your config says `save_every: 20`, weights are pushed every 500 iterations. The factor of 25 is hardcoded.
- **`online_update` is a third boolean on `component_build_args[name]`**, alongside `init` and `freeze`. Only components with `online_update: true` (and not `freeze: true`) have their weights pushed to the policy state manager. The Q-function in `resfit` is the canonical example of a component that *does* get optimized but should *not* be pushed to inference — see [`experiment_training/reinforcement_learning/resfit/online_rl/resfit.yaml`](../experiment_training/reinforcement_learning/resfit/online_rl/resfit.yaml).
- The state-dict is moved to **CPU** before `ray.put` so it lands in shared-memory plasma rather than tying up a GPU.

## Known oddities

These behaviors are documented faithfully because they are present in the code, not because they look intentional. Verify with the maintainer before relying on them.

- **wandb run name** is set via `name=f"{getattr(config.train, f'{project_name}', 'imitation_learning')}"`. This treats `task_name` (e.g. `"picknplace"`) as an *attribute name* on the `TrainConfig`. Unless your config genuinely defines a `picknplace:` field at `train:` level, this falls back to the literal string `"imitation_learning"`. The offline trainer uses `name=config.train.project_name` instead, which is the more useful behavior.
- **`epoch` in the online loop** advances only when `next(offline_iter)` raises `StopIteration`. It's used to label the checkpoint directory (`epoch_<N>/`), but two saves can land in the same `epoch_<N>/` if both happen during a single pass through the offline dataset (because save cadence is by iteration, not by epoch). Existing files get overwritten.
- **`num_iter_per_epoch = float(len(dataloader))`** is the offline dataloader's length. The `'epoch'` field logged to wandb (`iterations / num_iter_per_epoch`) is therefore a "synthetic epoch count" — useful as a proxy, misleading if you compare against an offline run.
