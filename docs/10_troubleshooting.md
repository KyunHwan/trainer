# Troubleshooting

A decision tree of failure modes. Each entry is **symptom → likely cause → fix → where to verify**.

## Contents

- [Launch failures](#launch-failures)
- [Config validation failures](#config-validation-failures)
- [Registry / plugin failures](#registry--plugin-failures)
- [Runtime CUDA / memory failures](#runtime-cuda--memory-failures)
- [DDP-specific failures](#ddp-specific-failures)
- [Data and stats failures](#data-and-stats-failures)
- [Ray online trainer failures](#ray-online-trainer-failures)
- [Save / resume failures](#save--resume-failures)

## Launch failures

### `AssertionError: LOCAL_RANK missing; launch with torchrun.`

**Cause**: `WORLD_SIZE` is set in the environment (> 1) but you launched with `python` directly.

**Fix**: either use `torchrun --nproc_per_node=N` for multi-GPU, or `unset WORLD_SIZE` before running for single-GPU. The assertion lives in `train()` in [`trainer/offline_trainer.py`](../trainer/offline_trainer.py).

### `ModuleNotFoundError: No module named 'model_constructor'`

**Cause**: the `policy_constructor` submodule isn't populated.

**Fix**:

```bash
git submodule update --init --recursive
ls policy_constructor/model_constructor   # should be non-empty
```

The trainer also auto-adds `policy_constructor/` to `sys.path` in [`trainer/modeling/factories.py`](../trainer/modeling/factories.py) if `model_constructor` isn't importable, so the directory just needs to exist with the right contents.

### `FileNotFoundError: Config file not found: ...`

**Cause**: the `--train_config` argument doesn't point at an existing YAML, or a `defaults:` entry references a missing file.

**Fix**: check the path. If it's a `defaults:` issue, the `ConfigLoadError` message includes the resolved path that wasn't found.

## Config validation failures

### `ConfigError: Config validation failed with N error(s): ...`

**Cause**: your YAML has a field that violates a Pydantic validator, or is missing a required field.

**How to read it**: every error has an `error_path` (dotted YAML path) and an `error_message`. Common ones:

- `error_path: model.find_unused_parameters; error_message: Field required` → add the field to the YAML, it has no default
- `error_path: model.component_config_paths; error_message: component_config_paths must contain at least one entry` → list at least one `name: path` pair
- `error_path: train.trainer.type; error_message: type must be a non-empty string` → fill in the registry key

**Fix**: edit the YAML and re-run. The schema reference is in [05_configuration.md § Pydantic schema](05_configuration.md#pydantic-schema).

### `ConfigLoadError: Config defaults cycle detected: A.yaml -> B.yaml -> A.yaml`

**Cause**: file A's `defaults:` lists B, and B's `defaults:` lists A (directly or transitively).

**Fix**: break the cycle. Defaults form a tree, not a graph. The cycle message shows the full include stack.

## Registry / plugin failures

### `KeyError: trainer registry has no key '<key>'. Available: [...]`

**Cause**: the module that registers `<key>` is missing from your YAML's `plugins:` list — or you typo'd the key, or the registering module has an import error that silently dropped it.

**Fix checklist**:

1. Grep the codebase for `register("<key>")` — find the file declaring it.
2. Confirm that file's module path is in `plugins:`. Example: a trainer in `experiment_training/components/trainer/imitation_learning/vfp_single_expert/vfp_single_expert_trainer.py` is referenced as `"experiment_training.components.trainer.imitation_learning.vfp_single_expert.vfp_single_expert_trainer"`.
3. Try importing the module manually: `python -c "import experiment_training.components.trainer.imitation_learning.vfp_single_expert.vfp_single_expert_trainer"`. If this fails with anything other than the import succeeding silently, fix that error.
4. Confirm the spelling of the `type:` value in YAML exactly matches the registered key. Registry keys are case-sensitive.

The same applies to `dataset_builder registry has no key '<key>'`, `loss_builder registry has no key '<key>'`, `optimizer_builder registry has no key '<key>'`.

### `KeyError: <name> registry already has key '<key>'`

**Cause**: the same module is being imported twice, or two different modules register the same key.

**Fix**: most often this means you have a duplicate `@TRAINER_REGISTRY.register("foo")` in two files. Pick one. The `_LOADED_MODULES` set in [`trainer/registry/plugins.py`](../trainer/registry/plugins.py) prevents the *same module path* from being imported twice via `plugins:`, but it doesn't help if two different modules use the same key.

### `TypeError: trainer registry expects subclasses of <Trainer protocol>, got <class>`

**Cause**: the class you decorated with `@TRAINER_REGISTRY.register(...)` doesn't structurally satisfy the `Trainer` protocol. Most commonly missing the `train_step` method, or `train_step` is not callable as a method.

**Fix**: add the missing method. See [04_concepts.md § Protocols (templates)](04_concepts.md#protocols-templates) for the required signatures.

## Runtime CUDA / memory failures

### `torch.cuda.OutOfMemoryError`

**Causes in priority order**:

1. **Per-rank batch too large**. Drop `data.batch_size`. Remember the global batch is `batch_size * world_size`.
2. **`prefetch_factor` is buffering too many batches**. Drop to `1` or remove the key.
3. **`find_unused_parameters: true` for an architecture that doesn't need it**. Set to `false` and rerun — saves memory and improves speed.
4. **Activation memory peaks during the largest forward**. Check for camera-pair concatenations (`torch.cat([head, left, right], dim=0)`) — those triple the per-batch image volume going through the backbone. Consider passing cameras one at a time.
5. **Mixed precision is on but a layer is keeping float32**. Most layers should be autocast-compatible under `bfloat16` (see [06_training_loop_walkthrough.md § Stage 10](06_training_loop_walkthrough.md#stage-10-per-batch-pipeline)). If a custom op forces float32, it bloats memory; rewrite if possible.

### `RuntimeError: NCCL communicator was aborted on rank N.`

**Cause**: typically a downstream effect of an OOM or assertion on one rank — the dead rank stops responding to collective ops and the others time out.

**Fix**: look at the rank-0 log first. Then check `dmesg | grep -i oom` for OOM-killed processes. Reduce memory pressure as above.

## DDP-specific failures

### Training hangs at a `dist.barrier()` or during the first epoch

**Cause**: control flow diverges across ranks. Common culprits:

- An exception inside the loop on one rank but not others (uncaught, kills the rank silently while peers wait for collective).
- A rank-0-only branch that includes a collective op (a `barrier`, an `all_reduce`).
- Different number of dataloader steps per rank (sampler `drop_last=False` mismatch, or differently-sized datasets per rank).

**Fix**: print the iteration count and rank from inside the loop on every rank, see who's stuck. If you added a branch like `if rank == 0: ...` recently, make sure nothing inside it calls a collective.

### `RuntimeError: Expected to have finished reduction in the prior iteration ...` (DDP)

**Cause**: a parameter that was used in the previous forward isn't used in the next one (or vice versa). DDP expects the set of "gradient-receiving" parameters to be the same across iterations.

**Fix**: set `model.find_unused_parameters: true`. This is the canonical fix for MoE-style models with input-dependent routing. The cost is a slower backward (DDP has to trace which parameters got gradients).

### Different metrics on different ranks after training

This shouldn't happen — only rank 0 logs. If you see two wandb runs being created, you probably have `wandb.init` somewhere outside the `if rank == 0:` gate. The framework calls `wandb.init` only on rank 0; anything else is custom code.

## Data and stats failures

### `dataset_stats.pkl` is missing after a run

**Cause**: the dataset factory didn't return a `'norm_stats'` key in its `build()` result.

**Fix**: check your factory. The contract is `return {"dataset": ..., "norm_stats": ...}` from `build()`. If you intentionally don't have stats, this is expected and the file simply isn't created — your trainer must then not assume `stats` contains anything.

Verify in [`_build_dataloader`](../trainer/offline_trainer.py): the pickle write is gated on rank 0 AND on the factory's returned dict containing a `'norm_stats'` key.

### `KeyError: 'task_name'` during `wandb.init`

**Cause**: `data.datamodule.params` doesn't include `task_name`.

**Fix**: add it:

```yaml
data:
  datamodule:
    type: "..."
    params:
      task_name: "my_task"
      ...
```

This becomes the wandb project name.

### Mixed-dtype error when concatenating online + offline batches

**Cause**: camera tensors arrive as `uint8` and weren't promoted to `float32` before concatenation. `cast_dtype` only touches floating tensors, so `uint8` slips through.

**Fix**: the online trainer already has a per-camera-key branch that promotes `uint8` to float before `Resize` and `cat`. If you're hitting this in a custom trainer, replicate the same promotion. The relevant code is around `if offline_data[key].dtype == torch.uint8: offline_data[key] = offline_data[key].float()` in [`online_trainer.py`](../trainer/online_trainer.py).

## Ray online trainer failures

### Online trainer hangs at `Going into getting buffer...` forever

**Cause**: the named `replay_buffer` actor exists but never accumulates `batch_size * 2 * world_size` samples. The trainer polls `.size.remote()` every 0.5s.

**Fix**: check that whatever populates the buffer is actually running. If you scaled `batch_size` up, the threshold (`batch_size × 2 × world_size`) may be larger than your collector can feed; either reduce the batch or accept a longer warmup.

### `ValueError: Actor with name 'replay_buffer' not found`

**Cause**: the named actor doesn't exist in the Ray cluster, or you're connected to the wrong Ray cluster.

**Fix**: `ray list actors` and confirm a `replay_buffer` actor is `ALIVE`. Same for `policy_state_manager`.

### Saved weights aren't appearing on inference workers

**Cause checklist**:

1. Save cadence is `(iterations + 1) % (save_every * 25) == 0` — confirm enough iterations have passed.
2. Only components with `component_build_args[name].online_update: true` (and `freeze: false`) get pushed. Components without that flag stay local. Check your YAML.
3. The push is `policy_state_manager.update_state.remote(weights_ref)` — it's a fire-and-forget. If the actor's `update_state` raises, the trainer never knows. Check the actor's log.

## Save / resume failures

### `<load_dir>/<name>.pt doesn't exist as a file!` warnings at startup

**Cause**: you set `train.load_dir` but the directory doesn't have a `<name>.pt` for every component in `model.component_config_paths`. The trainer is doing a partial load (see [08_checkpoints_and_resume.md § Partial loads](08_checkpoints_and_resume.md#partial-loads)).

**Action**: if intentional, fine — those components will use whatever weights `build_model` produced (or `init_weights` if `init: true` AND `load_dir is None`, which is not the case here). If unintentional, double-check the path and file names.

### Resume produces wildly different loss curves than expected

**Causes**:

- **Optimizer state not loaded** — `<name>_opt.pt` missing. The optimizer restarts from zero state (no momentum, no LR-schedule progress), which often spikes the loss in the first few steps.
- **Schedule progress reset** — for optimizers like `adamw_warmup_cosine_decay` that bundle a scheduler, the scheduler state lives inside the optimizer's `state_dict()` under `"scheduler"`. If you load model weights but not optimizer state, the LR schedule restarts from step 0.
- **Different config than the original run** — verify your `train.epoch`, batch size, LR, etc. match.

**Fix**: include both `<name>.pt` and `<name>_opt.pt` in `load_dir`.
