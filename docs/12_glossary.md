# Glossary

A reference of every term that appears in this codebase that a junior engineer might not know. Alphabetical.

### action horizon
The number of future action steps the policy predicts in a single forward pass. Concretely, `action_horizon: 40` in `data.datamodule.params` means each training example's `data['action']` tensor has shape `(B, 40, action_dim)` and the model is trained to predict the next 40 actions given the current observation.

### autocast (`torch.autocast`)
A context manager that automatically casts eligible operations to a lower-precision dtype during a forward pass. The framework uses `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` around `trainer.train_step`. See [bfloat16](#bfloat16).

### base_policy_action
An optional key in online-trainer data dicts. When present in the online batch (signaling a residual-policy setup like ResFit), the offline batch's `action` is copied into a new `base_policy_action` key so that the concatenated batch has the same keys on both sides.

### bfloat16
A 16-bit floating-point format with the same exponent range as float32 but ~3 decimal digits of precision. Used here for mixed-precision training under `torch.autocast`. Unlike float16, bfloat16 doesn't need `GradScaler` because the larger exponent range doesn't lose gradient magnitudes to underflow.

### CFG (classifier-free guidance)
A technique for conditional generative models in which both conditional and unconditional predictions are computed and combined at inference time. The `cfg_vqvae_flow_matching` experiment trains a model with random conditioning dropout so it learns both forms.

### CondOTScheduler
Conditional optimal-transport scheduler from the `flow_matching` library. Combined with `AffineProbPath` to define the noise → data interpolation used by every flow-matching trainer in this codebase.

### DDP (`DistributedDataParallel`)
PyTorch's standard data-parallel training wrapper. Each rank holds a full model replica; gradients are all-reduced across replicas after backward; each rank then applies the averaged update independently. See [02_distributed_training.md](02_distributed_training.md).

### DistributedSampler
A PyTorch sampler that, in a DDP setup, gives each rank a disjoint slice of the dataset every epoch. Must be paired with `shuffle=False` on the `DataLoader` (shuffling lives in the sampler) and `sampler.set_epoch(epoch)` at the top of each epoch (to vary the per-epoch permutation).

### EMA (exponential moving average)
A weighted-average of a parameter set across time: `ema_param = decay * ema_param + (1 - decay) * current_param`, applied step-by-step. The `EMAConfig` Pydantic model declares fields for it, but no current trainer implements EMA — declared but unused.

### find_unused_parameters
A DDP option (`DDP(..., find_unused_parameters=True)`) that traces every backward pass to detect parameters with no gradient and excludes them from the all-reduce. Needed when parameter usage depends on the input (e.g., mixture-of-experts routing). Adds backward-pass overhead.

### flow matching
A family of generative-model training objectives in which the model learns the *velocity field* of a probability path between noise and data. At training time, sample `t ~ Beta(1.0, 1.5)`, interpolate `x_t = (1-t) * noise + t * data`, and train the model to predict `dx_t = data - noise`. See [`AffineProbPath`](../experiment_training/components/trainer/imitation_learning/vfp_single_expert/vfp_single_expert_trainer.py) usage.

### GraphModel
The `nn.Module` type produced by `policy_constructor.build_model`. It's a structured DAG of submodules whose topology comes from a YAML config. See [policy_constructor's README](../policy_constructor/README.md).

### imitation learning
Training a policy by minimizing the discrepancy between policy predictions and recorded expert demonstrations. The opposite of reinforcement learning, where the policy learns by trying actions and observing rewards.

### `init_process_group`
PyTorch call (`torch.distributed.init_process_group(backend="nccl", init_method="env://")`) that establishes the inter-rank communication group. Required before any collective op. Called by `_dist_setup`.

### LeRobot
HuggingFace's robotics dataset / training library. This framework uses `lerobot.LeRobotDataset` as the data source for most experiments — see [`lerobot_data.py`](../experiment_training/components/dataloader/lerobot_data.py). `LeRobotDataset` understands `delta_timestamps` for action chunks and observation histories.

### MoE (mixture of experts)
An architecture in which several "expert" subnetworks compete; a gating network routes each input to a subset of experts whose outputs are then combined. Sparse routing reduces compute per input but requires `find_unused_parameters: true` under DDP.

### named actor (Ray)
A Ray actor (long-lived stateful worker) that is named at creation time and can be looked up later by name with `ray.get_actor("name")`. The online trainer relies on two: `replay_buffer` and `policy_state_manager`.

### NCCL
NVIDIA Collective Communications Library — the GPU-aware communication backend PyTorch DDP uses for `all_reduce`, `broadcast`, etc. The framework hardcodes `backend="nccl"` in `init_process_group`.

### `nn.ModuleDict`
A PyTorch container that holds an ordered mapping of named submodules. `parameters()` walks all of them; `state_dict()` includes all of them. The framework uses one to hold the named model components per experiment.

### norm_stats
Normalization statistics — a dict of `{key: {"mean": tensor, "std": tensor}}` returned by a dataset factory under the `"norm_stats"` key. Pickled to `dataset_stats.pkl` and passed into `trainer.train_step` as `stats=`. The framework does not apply them; the trainer decides whether and where to normalize.

### optimal transport (OT)
A family of distance metrics between probability distributions. This codebase uses the Sinkhorn-Knopp algorithm (entropic-regularized OT) via the `geomloss` library for the K-OT loss in VFP-style trainers.

### plugin
In this codebase, a plugin is a Python module listed in a YAML's `plugins:` field. Importing the module triggers its `@register` decorators, which populate the global registries. See [04_concepts.md § Plugins](04_concepts.md#plugins).

### policy_constructor
A YAML-driven model construction library, vendored as a git submodule. Builds `GraphModel` instances from declarative configs without containing any training logic. See [`policy_constructor/README.md`](../policy_constructor/README.md).

### Protocol (PEP 544)
A Python typing construct for *structural* subtyping — any class with the right methods satisfies the protocol, without explicit inheritance. The framework uses `@runtime_checkable` protocols so `isinstance(obj, Protocol)` can verify conformance at startup.

### Pydantic v2
The data-validation library powering this codebase's config schema. `BaseModel` declares typed fields with validators; `model_validate(raw)` runs validation; `ValidationError` is raised on failure. `ConfigDict(extra="allow")` lets unknown YAML keys pass through. `RootModel[X]` wraps a single value type as a model.

### Ray actor
A Ray construct: a long-lived Python object whose methods are invoked remotely. Method calls return `ObjectRef`s; you `ray.get(...)` to materialize the result.

### Ray Train
Ray's training-orchestration layer. Wraps your training function (`train_func`) and distributes it across cluster workers, sets up the process group, prepares dataloaders. Replaces `torchrun` + raw DDP for online jobs.

### `ray.put` / `ray.get`
Ray's object-store API. `ray.put(value)` writes a Python object into the cluster's plasma store and returns an `ObjectRef`; `ray.get(ref)` materializes the value (on the same or a different node). Used by the online trainer to broadcast weights without serializing them across every actor call.

### registry
A typed `dict[str, type]` with a decorator API for registration. See [04_concepts.md § Registries](04_concepts.md#registries). The framework has four global registries (trainer, dataset, optimizer, loss).

### RootModel
A Pydantic v2 model that wraps a single value rather than a struct of fields. The framework uses `RootModel[dict[str, str]]` for `ComponentConfigPaths` so the validator can run on the dict as a whole.

### `runtime_checkable`
A decorator applied to a Protocol. Without it, you can use the protocol for static type-checking only; with it, `isinstance(obj, Protocol)` works at runtime by checking method names.

### schedulefree
Aaron Defazio et al.'s schedule-free optimizer family ([github.com/facebookresearch/schedule_free](https://github.com/facebookresearch/schedule_free)). `RAdamScheduleFree` is wrapped by [`schedule_free_radam.py`](../experiment_training/components/optimizer/schedule_free_radam.py). No external learning-rate schedule needed.

### Sinkhorn-Knopp
An algorithm for entropic-regularized optimal transport. Iteratively computes scaling vectors that turn an exponentiated cost matrix into a doubly-stochastic transport plan. The `sinkhorn_knopp` loss uses `geomloss.SamplesLoss(loss="sinkhorn", ...)` for this.

### SyncBatchNorm
A drop-in BatchNorm replacement that computes running statistics over the *global* DDP batch via all-reduce, rather than over each rank's local slice. The framework calls `nn.SyncBatchNorm.convert_sync_batchnorm(...)` on any model containing BN layers when DDP is on.

### `task_index`
An integer column on each batch (`data['task_index']`) that indexes into a `tasks.parquet` table shipped with the LeRobot dataset, mapping each entry to a natural-language prompt. Trainers that need text conditioning look up the prompt from the parquet file at runtime.

### tokenization / `tokens.parquet`
*(Not currently used in this trainer.)* Some LeRobot-derived datasets pre-tokenize prompts into a `tokens.parquet` table. The trainers in this repo work with `task_index` and look up prompts only as needed.

### `torchrun`
PyTorch's distributed launcher. Replaces the older `python -m torch.distributed.launch`. Sets `WORLD_SIZE`, `LOCAL_RANK`, `RANK`, `MASTER_ADDR`, `MASTER_PORT`. Invocation: `torchrun --nproc_per_node=4 trainer/offline_trainer.py --train_config ...`.

### VFP (variational flow policy)
A flow-matching policy architecture with a variational bottleneck (a VQVAE-style posterior/prior pair) and optionally a mixture-of-experts decoder. "VFP single-expert" is the simpler variant without MoE. See [`experiment_training/components/trainer/imitation_learning/`](../experiment_training/components/trainer/imitation_learning/).
