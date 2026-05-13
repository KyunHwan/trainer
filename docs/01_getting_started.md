# Getting started

*Estimated reading time: 12 minutes*

This guide takes you from a fresh clone to a running single-GPU training job. You should not need to read anything else first.

## Contents

- [Prerequisites](#prerequisites)
- [Clone with submodules](#clone-with-submodules)
- [Set up the Python environment](#set-up-the-python-environment)
- [Pick a config](#pick-a-config)
- [Run training on one GPU](#run-training-on-one-gpu)
- [What success looks like](#what-success-looks-like)
- [What's on disk after a run](#whats-on-disk-after-a-run)
- [First failure modes](#first-failure-modes)
- [Next steps](#next-steps)

## Prerequisites

- **Python**: 3.10+ (developed with 3.12). The `uv_setup.sh` script creates a `.venv` and `env_setup.sh` installs into it.
- **GPU**: One CUDA-capable GPU is enough for this guide. The default config in this guide assumes ~24 GB of VRAM (batch size 60, 3-camera VFP single-expert). If you have less, drop `data.batch_size` in the YAML.
- **Disk**: ~30 GB for PyTorch + LeRobot + vision deps + a small HuggingFace dataset cache.
- **Weights & Biases**: Training logs to wandb by default. Either run `wandb login` once or set `WANDB_MODE=offline` before launching. There is no command-line flag to disable wandb in the current code.

## Clone with submodules

`policy_constructor` is a git submodule. Clone with `--recursive` so it's populated:

```bash
git clone --recursive <repo-url> trainer
cd trainer
```

If you already cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

Verify the submodule is populated:

```bash
ls policy_constructor/model_constructor   # should list config/, blocks/, registry/, ...
```

## Set up the Python environment

Two scripts. Run them from the repo root in order:

```bash
bash uv_setup.sh        # installs uv (if needed), creates .venv in repo root
source .venv/bin/activate
bash env_setup.sh       # installs every dependency into .venv via uv pip install
```

What `env_setup.sh` installs (groups roughly mirror the script — see [`env_setup.sh`](../env_setup.sh)):

- **PyTorch stack**: `torch==2.9.0`, `torchvision==0.24.0` (CUDA 13.0 wheels), `torchcodec==0.9.1`, `av`
- **Flow-matching / OT**: `flow_matching`, `schedulefree`, `geomloss`
- **Vision/utility**: `einops`, `timm`, `wandb`, `tqdm-loggable`
- **OpenPI compatibility shims**: `transformers==4.53.2`, `flax`, `augmax`, `beartype`, `jaxtyping==0.2.34`, `sentencepiece`, `chex`, `numpydantic`
- **Data**: `lerobot` (no-deps install), `datasets`, `accelerate`, `gcsfs`, plus system `ffmpeg` and `libav*` for video decoding
- **Misc**: `tyro`, `ml_collections`, `pytest`
- **Editable install** of Depth Anything v3 from inside the `policy_constructor` submodule

If you plan to train OpenPI-based experiments, also run [`openpi_transformer_lib_patch.sh`](../openpi_transformer_lib_patch.sh) after activation. It overwrites the `transformers` package files in `.venv/lib/python3.12/site-packages/transformers/` with patched versions shipped under `policy_constructor/.../transformers_replace/`. The VFP single-expert config used in this guide does **not** need the patch.

## Pick a config

Use the smallest fully-working training YAML in the repo, [`experiment_training/imitation_learning/vfp_single_expert/exp1/vfp_single_expert.yaml`](../experiment_training/imitation_learning/vfp_single_expert/exp1/vfp_single_expert.yaml). It is a complete, validated config — every field referenced by the trainer is present.

Two fields are worth checking before you run:

- `data.datamodule.params.repo_id` — by default it pulls a HuggingFace LeRobot dataset (`joon001001/igris-b-pnp-v4.1`). If the dataset isn't accessible, training will fail at dataset construction. Replace with a `repo_id` you have access to, or set `local_files_only: true` and point `root:` at a local LeRobot dataset.
- `train.save_dir` — defaults to a relative path inside the repo. Change to an absolute path (or `~/...`) if you'd rather keep checkpoints outside the working tree. The directory is created if missing.

## Run training on one GPU

From the repo root, with the venv activated:

```bash
python trainer/offline_trainer.py \
  --train_config experiment_training/imitation_learning/vfp_single_expert/exp1/vfp_single_expert.yaml
```

That's it. `WORLD_SIZE` defaults to 1 so DDP is not initialized; the trainer runs the loop on a single GPU. For multi-GPU, see [02_distributed_training.md](02_distributed_training.md).

## What success looks like

Within the first ~30 seconds you should see, in order:

1. **wandb init banner**. Includes a URL like `View run <name> at: https://wandb.ai/<entity>/<task_name>`. The wandb project is `data.datamodule.params.task_name` (e.g. `picknplace`); the run name is `train.project_name`.
2. **Per-component parameter counts**, one line per model in `model.component_config_paths`:
   ```text
   Parameters of head_backbone model: 313.4 M
   Parameters of left_backbone model: 313.4 M
   Parameters of right_backbone model: 313.4 M
   Parameters of info_embedder model: 12.8 M
   Parameters of action_decoder model: 22.6 M
   Total Parameters: 975.6 M
   ```
   (Numbers vary with config; what matters is that **every** component listed in the YAML appears.)
3. **`Global batch size = 60`** for single-GPU (with `data.batch_size: 60`). For multi-GPU this scales linearly with the number of ranks.
4. **A tqdm progress bar** for the first epoch. Each step is one optimizer update.
5. After `train.save_every` epochs: `Saved checkpoints for epoch <N> at <save_dir>/epoch_<N>`.

If you don't see all five, jump to [First failure modes](#first-failure-modes) below.

## What's on disk after a run

After at least `save_every` epochs (default 3) have completed, this is the layout of `train.save_dir`:

```text
<save_dir>/
├── dataset_stats.pkl                  # normalization statistics (pickled dict)
└── epoch_<N>/
    ├── <component>.pt                 # one per entry in model.component_config_paths
    └── <component>_opt.pt             # one per entry in model.component_optims
```

- `dataset_stats.pkl` is written **once**, by rank 0, the first time the dataloader is built. It's the value returned by the dataset factory under the `norm_stats` key.
- `<component>.pt` is the unwrapped (`DDP.module`) `state_dict()` of that named model. The file name does **not** include the epoch — the epoch is in the parent directory name.
- `<component>_opt.pt` is the optimizer `state_dict()`. For optimizers that bundle a scheduler (e.g. `adamw_warmup_cosine_decay`), the scheduler state lives under the `"scheduler"` key inside the optimizer state dict.

Frozen components have no `_opt.pt` (no optimizer was built for them).

To resume from one of these checkpoints, set `train.load_dir` to the `epoch_<N>` directory in your YAML. See [08_checkpoints_and_resume.md](08_checkpoints_and_resume.md).

## First failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: model_constructor` | Submodule not initialized | `git submodule update --init --recursive` |
| `wandb: ERROR Network error` or wandb prompts for an API key | Not logged in / no API key | `wandb login` or `export WANDB_MODE=offline` |
| `KeyError: trainer registry has no key 'vfp_single_expert_trainer'` | A `plugins:` entry is missing or misspelled | Confirm the YAML's `plugins` list includes every component module referenced by `type:` keys. See [10_troubleshooting.md](10_troubleshooting.md) |
| `CUDA out of memory` | Batch too large for your GPU | Drop `data.batch_size` and/or `data.prefetch_factor` |
| `LOCAL_RANK missing; launch with torchrun.` | `WORLD_SIZE > 1` in env but you ran `python ...` directly | Use `torchrun --nproc_per_node=N trainer/offline_trainer.py ...` (or `unset WORLD_SIZE`) |
| Dataset download stalls or fails | HuggingFace cache misconfigured / no network | Set `data.datamodule.params.local_files_only: true` and pre-populate `root:` with a local LeRobot dataset |

A larger decision tree is in [10_troubleshooting.md](10_troubleshooting.md).

## Next steps

- **Scale out to multiple GPUs** — [02_distributed_training.md](02_distributed_training.md)
- **Understand what's happening inside `train()`** — [06_training_loop_walkthrough.md](06_training_loop_walkthrough.md)
- **Write your own trainer, loss, or dataset** — [07_extending.md](07_extending.md)
- **Look up a YAML field** — [05_configuration.md](05_configuration.md)
