# Experiment tracking (wandb)

## Contents

- [Conventions](#conventions)
- [What gets logged](#what-gets-logged)
- [Rank-0 gating](#rank-0-gating)
- [Running offline / disabling wandb](#running-offline--disabling-wandb)
- [Online trainer differences](#online-trainer-differences)

## Conventions

The offline trainer initializes wandb once on rank 0:

```python
project_name = config.data.datamodule.params["task_name"]
wandb.init(
    project=project_name,
    config=_params_dict(config),
    name=config.train.project_name,
)
```

| wandb field | Source |
|---|---|
| `project` | `config.data.datamodule.params["task_name"]` — e.g. `"picknplace"` |
| `name` (run name) | `config.train.project_name` — e.g. `"vfp_single_expert_adamw_cosine_1e-4_v3.3.2"` |
| `config` | The whole `ExperimentConfig` flattened via `_params_dict` |

If `task_name` is missing from the datamodule params, `wandb.init` raises `KeyError` immediately. Adding the field is straightforward — see existing configs in [`experiment_training/imitation_learning/`](../experiment_training/imitation_learning/) for examples.

The run name is a string under your control. It's the human-readable label that appears in the wandb UI alongside the auto-generated run ID.

## What gets logged

Every iteration, rank 0 calls `_record(loss_dict, iterations, num_iter_per_epoch)`:

```python
def _record(loss_dict, iterations, num_iter_per_epoch):
    detached_loss = {}
    for key in loss_dict.keys():
        if isinstance(loss_dict[key], torch.Tensor):
            if loss_dict[key].device.type == 'cpu':
                detached_loss[key] = loss_dict[key].item()
            else:
                detached_loss[key] = loss_dict[key].detach().item()
        else:
            detached_loss[key] = loss_dict[key]
    detached_loss['epoch'] = iterations / num_iter_per_epoch
    wandb.log(detached_loss, step=iterations)
```

Three things to know:

1. **The shape of `loss_dict`** is whatever your trainer's `train_step` returned. Keys become metric names; values are detached and converted to Python scalars. Non-tensor values pass through unchanged.
2. **`step=iterations` is the global step**, incremented on every rank's iteration (see [02_distributed_training.md § Rank-0-only side effects](02_distributed_training.md#rank-0-only-side-effects)). It is *not* the wandb-internal step counter — wandb is told explicitly what step this is.
3. **`'epoch'` is appended**. It's a float (`iterations / num_iter_per_epoch`), useful for plotting "loss vs fractional epoch".

A typical `loss_dict` from one of the imitation-learning trainers looks like:

```python
{
    "total":                    tensor(0.12, device="cuda:0"),
    "velocity":                 0.118,
    "Sinkhorn":                 0.0021,
    "info_embedder grad_norm":  1.34,
    "info_embedder lr":         9.5e-05,
    "action_decoder grad_norm": 0.87,
    "action_decoder lr":        9.5e-05,
}
```

After `_record` runs, all of these become wandb metrics under the same names, with `epoch` added.

## Rank-0 gating

`wandb.init`, every `wandb.log` call (via `_record`), and `wandb.finish` are all rank-0-only. Other ranks make no wandb calls. This is enforced by the `if rank == 0:` checks around the calls — no rank-awareness inside wandb is needed.

If you add custom logging inside `train_step`, **do not call `wandb.log` directly**. Either return your metric in the loss_dict (preferred — survives the rank-0 gating automatically), or wrap your call in `if torch.distributed.get_rank() == 0:` yourself. Otherwise non-rank-0 processes will either crash (no `wandb.init` was called there) or, worse, succeed and duplicate-log.

## Running offline / disabling wandb

There is no command-line flag in the trainer to disable wandb. Use the standard wandb env-var:

```bash
export WANDB_MODE=offline           # logs to local directory, no network
# or
export WANDB_MODE=disabled          # no logging at all
```

`offline` writes to `./wandb/` (gitignored). You can sync later with `wandb sync wandb/offline-run-*`.

Other useful env-vars:

| Variable | Effect |
|---|---|
| `WANDB_ENTITY=<team-or-user>` | Override the wandb entity (default: your default entity) |
| `WANDB_PROJECT=<name>` | Override the auto-derived project. The trainer passes `project` explicitly, but `WANDB_PROJECT` is what wandb uses if nothing is passed |
| `WANDB_RUN_GROUP=<group>` | Group multiple ranks/runs together (useful if you script multi-seed sweeps) |

## Online trainer differences

[`online_trainer.py`](../trainer/online_trainer.py) uses the same `wandb.init` pattern but with a different `name`:

```python
name=f"{getattr(config.train, f'{project_name}', 'imitation_learning')}",
```

This is unusual — it looks up `task_name` (e.g. `"picknplace"`) as an *attribute name* on `config.train`, falling back to the literal string `"imitation_learning"`. Unless your YAML declares a key matching the task name at the `train:` level (which none of the current configs do), every online run will be named `"imitation_learning"` in wandb.

> TODO (maintainer): this looks like a typo/bug — most likely intended `config.train.project_name`, mirroring the offline trainer. Confirm and fix at the source, then this section can be deleted.

The metric dict, `_record` function, and rank-0 gating are otherwise identical to the offline trainer. The only other functional difference is that `'epoch'` (= `iterations / num_iter_per_epoch`) divides by the *offline* dataloader length, so it's a proxy at best — not a true notion of progress through the live data.
