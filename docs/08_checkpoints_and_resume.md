# Checkpoints and resume

## Contents

- [Disk layout](#disk-layout)
- [Save semantics](#save-semantics)
- [Resume semantics](#resume-semantics)
- [The init and freeze flags during resume](#the-init-and-freeze-flags-during-resume)
- [Partial loads](#partial-loads)
- [Online trainer cadence](#online-trainer-cadence)

## Disk layout

Given `train.save_dir: /path/to/runs/my_experiment`, after several epochs with `train.save_every: 3`:

```text
/path/to/runs/my_experiment/
├── dataset_stats.pkl
├── epoch_3/
│   ├── head_backbone.pt
│   ├── head_backbone_opt.pt
│   ├── left_backbone.pt
│   ├── left_backbone_opt.pt
│   ├── right_backbone.pt
│   ├── right_backbone_opt.pt
│   ├── info_embedder.pt
│   ├── info_embedder_opt.pt
│   ├── action_decoder.pt
│   └── action_decoder_opt.pt
├── epoch_6/
│   └── ... (same files) ...
├── epoch_9/
│   └── ... (same files) ...
└── ...
```

Three rules:

- The file names do **not** include the epoch number. The epoch is the directory name.
- `<component>.pt` is the model `state_dict()` (DDP-unwrapped).
- `<component>_opt.pt` is the optimizer `state_dict()`. For optimizers that bundle a scheduler, the scheduler state is nested inside under `"scheduler"`.
- Frozen components have no `_opt.pt` (no optimizer was built for them).
- `dataset_stats.pkl` is written once on the first dataloader build by rank 0. It is **not** epoch-versioned.

## Save semantics

`_save_checkpoints` in [`trainer/offline_trainer.py`](../trainer/offline_trainer.py):

```python
def _save_checkpoints(models, optimizers, save_dir, epoch):
    epoch_folder = os.path.join(save_dir, f"epoch_{epoch}")
    os.makedirs(epoch_folder, exist_ok=True)
    for key in models.keys():
        state_to_save = models[key].module.state_dict() if isinstance(models[key], DDP) \
                                                        else models[key].state_dict()
        torch.save(state_to_save, os.path.join(epoch_folder, f"{key}.pt"))
        if key in optimizers:
            torch.save(optimizers[key].state_dict(),
                       os.path.join(epoch_folder, f"{key}_opt.pt"))
```

Called only from rank 0:

```python
if rank == 0 and (epoch + 1) % config.train.save_every == 0:
    _save_checkpoints(..., epoch=epoch + 1)
```

Epochs are **1-indexed in directory names** (`epoch_3/` is what you get after the third epoch finishes), even though Python's `range(epoch)` is 0-indexed.

DDP unwrap is unconditional — `models[key].module.state_dict()` is used whenever the entry is a `DDP` instance. This means saved `.pt` files are always plain `state_dict`s that load directly into an un-wrapped `nn.Module`, regardless of how the run was launched.

There is no atomic-rename or write-to-tempfile machinery: `torch.save` writes directly. If the process is killed mid-save you may end up with a truncated `.pt`. In practice this rarely happens because each individual save is fast, but be aware.

## Resume semantics

Set `train.load_dir` in your YAML to the directory containing the checkpoint files (typically an `epoch_<N>/` directory):

```yaml
train:
  load_dir: "/path/to/runs/my_experiment/epoch_30"
```

What happens at startup, per component (in [`_build_models`](../trainer/offline_trainer.py)):

```python
if config.train.load_dir is not None:
    path = os.path.join(config.train.load_dir, f"{k}.pt")
    if os.path.isfile(path):
        policy.load_state_dict(torch.load(path, map_location='cpu'))
    else:
        print(f"{path} doesn't exist as a file!")
else:
    if config.model.component_build_args[k]['init']:
        policy.apply(init_weights)
```

Three things to notice:

1. **Loading happens on CPU** (`map_location='cpu'`), then the model is moved to its rank's GPU after potential `SyncBatchNorm` conversion and DDP wrap. This avoids cross-rank GPU memory contention during load.
2. **Loading is best-effort**. If `<load_dir>/<k>.pt` is missing, the trainer prints a warning and **does not** fall back to `init_weights` — the model keeps whatever weights `policy_constructor.build_model` produced (which itself may have pretrained weights baked in, depending on the component's YAML).
3. **`init: true` is only honored when `load_dir is None`**. If you set `load_dir`, the `init` flag is effectively ignored for any component that has a file there.

The optimizer-resume logic in [`_build_optimizers`](../trainer/offline_trainer.py) is symmetric:

```python
if config.train.load_dir is not None:
    path = os.path.join(config.train.load_dir, f"{model_name}_opt.pt")
    if os.path.isfile(path):
        optimizers[model_name].load_state_dict(torch.load(path, map_location=device))
    else:
        print(f"{path} doesn't exist as a file!")
```

Optimizers are loaded with `map_location=device` (per-rank GPU), which is correct because optimizer state already contains GPU tensors when the source process was on GPU.

## The init and freeze flags during resume

The interaction matrix:

| `load_dir` set? | File exists for component? | `init` | `freeze` | Result |
|---|---|---|---|---|
| no  | n/a | true  | false | Apply `init_weights` |
| no  | n/a | false | false | Keep whatever `build_model` produced (often pretrained weights baked in) |
| no  | n/a | any   | true  | Freeze; skip optimizer creation |
| yes | yes | any   | false | Load weights from disk (overrides `init`) |
| yes | yes | any   | true  | Load weights from disk, then freeze; skip optimizer creation |
| yes | no  | any   | false | Print warning, keep `build_model` defaults |
| yes | no  | any   | true  | Print warning, keep `build_model` defaults, then freeze |

The "init only fires when no checkpoint is loaded" pattern is intentional: resume jobs should not re-randomize their weights. If you want to *force* re-init even when a checkpoint exists, delete or rename that component's `.pt` file in `load_dir`.

## Partial loads

A natural consequence of "loading is best-effort per file": you can resume *some* components from a previous run and start *others* fresh. Curate which files exist in `load_dir`:

```text
epoch_30/
├── head_backbone.pt          # will be loaded
├── head_backbone_opt.pt      # will be loaded
├── info_embedder.pt          # will be loaded
├── info_embedder_opt.pt      # will be loaded
# action_decoder.pt deliberately omitted -> trainer will print warning and use
# either init_weights (if init: true) or build_model's defaults (if init: false)
```

This is the workflow for "swap out the action decoder, keep everything else" experiments.

If you do this, double-check the printed warnings at startup. The line is:

```text
<load_dir>/<name>.pt doesn't exist as a file!
```

If you see that line for a component you expected to resume, you've named the file wrong or the directory is wrong.

## Online trainer cadence

[`online_trainer.py`](../trainer/online_trainer.py) saves on a different cadence:

```python
if (iterations + 1) % (config.train.save_every * 25) == 0:
    _save_checkpoints(..., epoch=epoch)
```

- Cadence is **iterations × 25 × save_every**, not epochs.
- The `epoch` value passed to `_save_checkpoints` is the offline-loop's epoch counter, which advances only when the offline dataloader exhausts. So two saves can land inside the same `epoch_<N>/` directory if both happen within one pass of the offline dataset; later saves overwrite earlier ones.
- The disk layout is otherwise identical to the offline trainer.

Resume from an online run uses the same `train.load_dir` mechanism as offline. There is no separate "resume the replay buffer" mechanism — the buffer's contents are not checkpointed by this trainer.
