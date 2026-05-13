# trainer

A config-driven distributed training framework for imitation-learning and reinforcement-learning policies. Models are composed from YAML via [policy_constructor](policy_constructor/), training components are selected through a registry/plugin system, and the training loop scales from a single GPU (PyTorch DDP) to a Ray-coordinated online/offline hybrid.

## Architecture

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │                       YAML Experiment Config                        │
 │  (plugins, model, data, train)                                      │
 └───────────────┬──────────────────────────────────────────────────────┘
                 │  load_config()  +  validate_config()
                 ▼
 ┌───────────────────────────┐    ┌──────────────────────────────────┐
 │  trainer/config/loader.py │───▶│ trainer/config/schemas.py        │
 │  YAML with defaults       │    │ Pydantic: ExperimentConfig       │
 │  composition & deep merge │    │  ├─ ModelConfig                  │
 └───────────────────────────┘    │  ├─ DataConfig                   │
                                  │  └─ TrainConfig                  │
                                  └──────────────┬───────────────────┘
                                                 │
                 ┌───────────────────────────────┼───────────────────┐
                 │                               │                   │
                 ▼                               ▼                   ▼
 ┌───────────────────────┐   ┌──────────────────────┐  ┌────────────────────┐
 │  Plugin Loader         │   │  Model Factory        │  │  Registries         │
 │  registry/plugins.py   │   │  modeling/factories.py │  │  TRAINER_REGISTRY   │
 │  importlib.import →    │   │  PolicyConstructor     │  │  DATASET_BUILDER_   │
 │  register components   │   │  ModelFactory          │  │  OPTIMIZER_BUILDER_ │
 └───────────────────────┘   │  → build_model() per   │  │  LOSS_BUILDER_      │
                              │    component config    │  └────────────────────┘
                              └──────────┬─────────────┘
                                         │
                                         ▼
                              ┌──────────────────────┐
                              │  nn.ModuleDict        │
                              │  {name: GraphModel}   │
                              │  per-model freeze /   │
                              │  init / DDP wrap      │
                              └──────────┬────────────┘
                                         │
           ┌─────────────────────────────┼──────────────────────────┐
           │                             │                          │
           ▼                             ▼                          ▼
 ┌──────────────────┐     ┌────────────────────────┐   ┌─────────────────────┐
 │ DatasetFactory    │     │  Trainer (protocol)     │   │ OptimizerFactory    │
 │ → Dataset +       │     │  .train_step(data,      │   │ → Optimizer per     │
 │   norm_stats      │     │              stats)     │   │   model component   │
 └────────┬─────────┘     │  forward / backward /   │   └─────────────────────┘
          │                │  clip / step            │
          ▼                └────────────┬───────────┘
 ┌──────────────────────┐               │
 │ DataLoader            │               │
 │ DistributedSampler    │               │
 │ (DDP) or              │               │
 │ ray.train.torch       │               │
 │ .prepare_data_loader  │               │
 └────────┬──────────────┘               │
          │                              │
          ▼                              ▼
 ┌────────────────────────────────────────────────────────┐
 │                    Training Loop                        │
 │  cast_dtype → move_to_device                            │
 │  autocast(bfloat16) → trainer.train_step(data, stats)  │
 │  rank-0: wandb.log + _save_checkpoints                 │
 │  barrier → next epoch                                   │
 └────────────────────────────────────────────────────────┘
          │
          │  (online_trainer.py only)
          ▼
 ┌────────────────────────────────────────────────────────┐
 │  Ray Actors                                             │
 │  replay_buffer.sample.remote() → online data            │
 │  policy_state_manager.update_state.remote()             │
 │  → ray.put(cpu_state_dicts) for inference workers       │
 └────────────────────────────────────────────────────────┘
```

## Where to go next

| I want to... | Read |
|---|---|
| Run my first training job | [docs/01_getting_started.md](docs/01_getting_started.md) |
| Understand the architecture | [docs/04_concepts.md](docs/04_concepts.md) |
| Write a custom trainer/loss/dataset | [docs/07_extending.md](docs/07_extending.md) |
| Debug a crash | [docs/10_troubleshooting.md](docs/10_troubleshooting.md) |
| Look up a YAML field | [docs/05_configuration.md](docs/05_configuration.md) |
| Browse all docs | [docs/README.md](docs/README.md) |

## Prerequisites

- Python 3.10+ (developed with 3.12)
- CUDA-capable GPU(s) — single GPU is enough for the getting-started guide

## Quickstart

```bash
# 1. Clone with the policy_constructor submodule
git clone --recursive <repo-url> trainer
cd trainer

# 2. Create the virtual environment via uv, activate it, install deps
bash uv_setup.sh
source .venv/bin/activate
bash env_setup.sh

# 3. Run training on one GPU
python trainer/offline_trainer.py \
  --train_config experiment_training/imitation_learning/vfp_single_expert/exp1/vfp_single_expert.yaml
```

For multi-GPU: `torchrun --nproc_per_node=<N> trainer/offline_trainer.py --train_config <config.yaml>`.
For Ray online training: see [docs/03_ray_online_training.md](docs/03_ray_online_training.md).

A full step-by-step walkthrough — including expected output, on-disk artifacts, and the most common failure modes — is in [docs/01_getting_started.md](docs/01_getting_started.md).

## Repository layout

```
├── trainer/                     Core training framework
│   ├── offline_trainer.py       DDP offline training entrypoint
│   ├── online_trainer.py        Ray Train online/offline hybrid entrypoint
│   ├── config/                  YAML loader + Pydantic schemas
│   ├── modeling/                Model factory (policy_constructor adapter)
│   ├── registry/                Registry system + plugin loader
│   ├── templates/               Protocol definitions (Trainer, DatasetFactory, etc.)
│   └── utils/                   Device, tree, seed, selection, import helpers
│
├── experiment_training/         Experiment implementations (plugins)
│   ├── components/              Registered trainers, data loaders, losses, optimizers
│   ├── imitation_learning/      IL training YAML configs (5 algorithms × variants)
│   └── reinforcement_learning/  RL training YAML configs (resfit, dsrl_openpi)
│
├── experiment_models/           Model architecture configs (policy_constructor YAML)
│   ├── vfp_single_expert/
│   ├── variational_flow_matching_policy/
│   ├── naive_flow_matching_policy/
│   ├── cfg_vqvae_flow_matching/
│   ├── mutual_information_estimator/
│   ├── resfit/
│   ├── openpi_batched/
│   └── dsrl_openpi/
│
├── policy_constructor/          Model construction library (git submodule)
├── docs/                        Documentation hub (start at docs/README.md)
├── env_setup.sh                 Dependency installation script
├── uv_setup.sh                  Virtual environment setup
└── openpi_transformer_lib_patch.sh   Shim required for OpenPI experiments
```

## License

> TODO (maintainer): add a license file and link it here. The repository currently has no top-level `LICENSE` file.

## Contact

> TODO (maintainer): add a contact / maintainership line — owner, channel for questions, issue-tracker URL.
