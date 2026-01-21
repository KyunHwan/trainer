"""Public API entrypoint for training."""
from __future__ import annotations

import os
import gc
from pathlib import Path

from trainer.config.loader import load_config
from trainer.config.schemas import ExperimentConfig, validate_config
from trainer.config.schemas import OptimizerParams
from trainer.modeling.factories import PolicyConstructorModelFactory
from trainer.registry import (
    TRAINER_REGISTRY,
    DATASET_BUILDER_REGISTRY,
    OPTIMIZER_BUILDER_REGISTRY,
    LOSS_BUILDER_REGISTRY,
)

from trainer.templates import (
    DatasetFactory,
    LossFactory,
    OptimizerFactory,
    Trainer
)
from trainer.registry.plugins import load_plugins
from trainer.utils.import_utils import instantiate
from trainer.utils.seed import *
import argparse

from trainer.utils.device import move_to_device, cast_dtype
from trainer.utils.tree import tree_map

import ray
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler

from tqdm import tqdm
import wandb
import pickle

def _params_dict(params) -> dict:
    if hasattr(params, "model_dump"):
        return params.model_dump()
    return params

def init_weights(m):
    # Initialize Convolutional Layers
    if isinstance(m, nn.Conv2d):
        # Xavier (Glorot) is often good for Tanh/Sigmoid or general cases
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)
            
    # Initialize BatchNorm Layers
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

    elif isinstance(m, nn.Linear):
        # Kaiming (He) is best for ReLU activations
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # Biases are usually initialized to 0
        if m.bias is not None:
            init.constant_(m.bias, 0)

def map_list_to_torch(lst: list):
    import torch
    return torch.tensor(lst)



""" Distributed Training Helpers """

def sync(tag: str, local_rank):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if local_rank == 0:
        print(f"[sync ok] {tag}", flush=True)

def _model_fingerprint(m: nn.Module):
    import hashlib
    import json
    items = []
    for n, p in m.named_parameters():
        items.append((n, tuple(p.shape), str(p.dtype), p.requires_grad))
    # include buffers too (even if broadcast_buffers=False, mismatched buffers can indicate mismatched build)
    for n, b in m.named_buffers():
        items.append((f"BUF:{n}", tuple(b.shape), str(b.dtype), False))
    s = json.dumps(items, sort_keys=True).encode("utf-8")
    return hashlib.sha1(s).hexdigest(), items

def find_unregistered_tensors(m: nn.Module):
    param_ids = {id(p) for p in m.parameters()}
    buffer_ids = {id(b) for b in m.buffers()}
    weird = []
    for mod_name, mod in m.named_modules():
        for k, v in vars(mod).items():
            if torch.is_tensor(v) and id(v) not in param_ids and id(v) not in buffer_ids:
                weird.append((mod_name, k, str(v.device), tuple(v.shape), str(v.dtype)))
    return weird

def _dist_setup(enable_dist_train, device) -> None:
    if enable_dist_train:
        dist.init_process_group(backend="nccl",
                                init_method="env://",
                                )

def _dist_barrier(dist_enabled, local_rank) -> None:
    if dist_enabled:
        dist.barrier(device_ids=[local_rank])

def _dist_cleanup(enable_dist_train) -> None:
    if enable_dist_train:
        dist.destroy_process_group()



""" Training Pipeline Component Construction """

def _build_loss(config, device) -> nn.Module:
    """
    Required Config Params:
        train.loss.type
        train.loss.params
    """
    loss_cls = LOSS_BUILDER_REGISTRY.get(config.train.loss.type)
    loss_fn = instantiate(loss_cls, _params_dict(config.train.loss.params))
    loss_fn = loss_fn.build()
    return loss_fn if not isinstance(loss_fn, nn.Module) else loss_fn.to(device)

def _build_models(world_size, global_rank, local_rank, enable_dist_train, config, device) -> nn.ModuleDict[str, nn.Module]:
    """
    Required Config Params:
        model.config_path
        train.load_dir
    """
    project_dir = Path(__file__).resolve().parent.parent
    
    model_config_paths = config.model.component_config_paths.as_dict()
    for key in model_config_paths.keys():
        if model_config_paths[key] and not os.path.isabs(model_config_paths[key]):
            model_config_paths[key] = os.path.abspath(
                os.path.join(project_dir, model_config_paths[key])
            )

    model_factory = PolicyConstructorModelFactory()
    model = model_factory.build(model_config_paths)
    models = {"main": model} if not isinstance(model, dict) else model
    model_total_params = 0.0
    for k, policy in models.items():
        frozen = False
        if local_rank == 0: 
            n_params_m = sum(p.numel() for p in policy.parameters()) / 1000000.0
            model_total_params += n_params_m
            print(f"Parameters of {k} model: {n_params_m:.1f} M")

        # Load model checkpoints / initialization
        # Need to check that saved models weren't wrapped using DDP (ie. that they aren't wrapped using modules)
        if config.train.load_dir is not None:
            policy.load_state_dict(torch.load(os.path.join(config.train.load_dir, f"{k}.pt"), map_location='cpu'))
        else:
            if config.model.component_build_args[k]['init']: 
                policy.apply(init_weights)
        
        if config.model.component_build_args[k]['freeze']:
            frozen = True
            for param in policy.parameters():
                param.requires_grad_(False)
            policy.eval()

        # If BatchNorm layers in a real model, convert to SyncBatchNorm so BN stats sync across replicas
        # This should be before moving the model onto a device
        if torch.cuda.is_available() and enable_dist_train and any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in policy.modules()):
            policy = nn.SyncBatchNorm.convert_sync_batchnorm(policy)

        # If the model wasn't loaded onto device, load it onto device
        policy = policy.to(device)
        
        # 'find_unused_parameters' is used when there are conditional cases that leave some parts of the model unexplored. 
        # For example, when mixture of experts is used.
        find_unused_parameters = getattr(config.model, "find_unused_parameters", False)

        models[k] = DDP(policy, find_unused_parameters=find_unused_parameters) if enable_dist_train and not frozen else policy

    if local_rank == 0: 
        print(f"Total Parameters: {model_total_params:.1f} M")

    return nn.ModuleDict(models)

def _build_optimizers(config, models: nn.ModuleDict[str, nn.Module], device) -> dict[str, torch.optim.Optimizer]:
    """
    Required Config Params:
        train.optimizer.type
        train.optimizer.params
        train.load_dir
    """
    
    # One optimizer per model
    # Only build optimizer for a model that is registered to have a optimizer
    optimizers = {}
    for model_name in config.model.component_optims.keys():
        # Skip models that are frozen
        params = [p for p in models[model_name].parameters() if p.requires_grad]
        if len(params) == 0:
            continue 
        optimizer_cls = OPTIMIZER_BUILDER_REGISTRY.get(config.model.component_optims[model_name]['type'])
        optimizer_factory = instantiate(optimizer_cls, 
                                        _params_dict(OptimizerParams.model_validate(config.model.component_optims[model_name]['params']).model_dump()))
        optimizers[model_name] = optimizer_factory.build(models[model_name].parameters())
        if config.train.load_dir is not None:
            optimizers[model_name].load_state_dict(torch.load(os.path.join(config.train.load_dir, f"{model_name}_opt.pt"), map_location=device))
    
    return optimizers

def _build_dataloader(config, world_rank=0, local_rank=0, world_size=0, enable_dist_train=False):
    """
    Required Config Params:
        data.datamodule.type
        data.datamodule.params
        data.batch_size
        data.num_workers
        data.pin_memory
        data.persistent_workers 
        train.save_dir
    """
    
    datamodule_cls = DATASET_BUILDER_REGISTRY.get(config.data.datamodule.type)
    dataset_factory = instantiate(datamodule_cls, params=config.data.datamodule.params, config=config)
    returned_product = dataset_factory.build(
                            {
                                'local_rank': local_rank,
                                'dist_enabled': enable_dist_train,
                                'save_dir': config.train.save_dir
                            },
                            params=config.data.datamodule.params)

    if isinstance(returned_product, dict):
        for key in returned_product.keys():
            if key == "dataset":
                dataset = returned_product[key]
            elif key == 'norm_stats':
                stats = returned_product[key]
                if local_rank == 0:
                    try:
                        stats_path = os.path.join(config.train.save_dir, f"dataset_stats.pkl")
                        with open(stats_path, "wb") as f:
                            pickle.dump(stats, f)
                    except:
                        stats_path = Path(os.path.join(config.train.save_dir, f"dataset_stats.pkl")).expanduser()
                        with open(stats_path, "wb") as f:
                            pickle.dump(stats, f)
    else: 
        dataset = returned_product
        stats = None

    #repo_id = 'joon001001/igris-b-pnp-lerobot'
    #dataset = LeRobotDataset(repo_id)

    # When using DistributedSampler, do NOT set shuffle=True on DataLoader.
    # Shuffling is handled by the sampler (see PyTorch DDP tutorial pattern)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=world_rank, drop_last=True) if enable_dist_train else RandomSampler(dataset)
    dataloader = ray.train.torch.prepare_data_loader(DataLoader(dataset, 
                            sampler=sampler, 
                            batch_size=config.data.batch_size,
                            num_workers=config.data.num_workers,
                            pin_memory=config.data.pin_memory,
                            persistent_workers=config.data.persistent_workers,
                            prefetch_factor=config.data.prefetch_factor,
                            worker_init_fn=seed_worker,
                            drop_last=False,
                            shuffle=False))
    return dataloader, sampler, stats

def _build_trainer(world_size, global_rank, local_rank, enable_dist_train, config, device) -> Trainer:
    """
    Required Config Params:
        train.trainer.type
        train.trainer.params
    """
    models = _build_models(world_size, global_rank, local_rank, enable_dist_train, config, device)
    optimizers = _build_optimizers(config, models, device)
    loss_fn = None
    if config.train.loss is not None:
        loss_fn = _build_loss(config, device)

    trainer_cls = TRAINER_REGISTRY.get(config.train.trainer.type)
    trainer = instantiate(
        trainer_cls,
        _params_dict(config.train.trainer.params),
        models=models,
        optimizers=optimizers,
        loss=loss_fn,
        device=device
    )
    if not isinstance(trainer, Trainer):
        raise TypeError("Constructed object does not match Trainer interface")
    return trainer


""" Parameter Saving """

def _save_checkpoints(models: nn.ModuleDict, 
                      optimizers: dict[str, torch.optim.Optimizer], 
                      save_dir: str, 
                      epoch: int):
    """
    Saves model and optimizer checkpoints to a folder named 'epoch' inside save_dir.
    
    File structure created:
        save_dir/
            epoch/
                {key}_{epoch}.pt
                {key}_opt_{epoch}.pt
    """
    # 1. Define the specific folder path "epoch" inside save_dir
    epoch_folder = os.path.join(save_dir, f"epoch_{epoch}")
    
    # 2. Create the directory if it doesn't exist (exist_ok=True acts as 'overwrite/update')
    os.makedirs(epoch_folder, exist_ok=True)
    
    # 3. Iterate through keys (assuming keys match as per instructions)
    for key in models.keys():
        # --- Save Model ---
        model_filename = f"{key}.pt"
        model_path = os.path.join(epoch_folder, model_filename)

        # Check if wrapped
        # Access the original model to save "clean" weights
        state_to_save = models[key].module.state_dict() if isinstance(models[key], DDP) else models[key].state_dict()

        torch.save(state_to_save, model_path)
        
        # --- Save Optimizer ---
        # We assume the key exists in optimizers as stated in the prompt
        if key in optimizers:
            opt_filename = f"{key}_opt.pt"
            opt_path = os.path.join(epoch_folder, opt_filename)
            
            torch.save(optimizers[key].state_dict(), opt_path)
            
    print(f"Saved checkpoints for epoch {epoch} at {epoch_folder}")


""" Training Info Logging """

def _record(loss_dict: dict[str, Any], iterations: int, num_iter_per_epoch: float): 
    detached_loss = {}

    for key in loss_dict.keys():
        if isinstance(loss_dict[key], torch.Tensor):
            if loss_dict[key].device.type == 'cpu':
                detached_loss[key] = loss_dict[key].item()
            else:
                detached_loss[key] = loss_dict[key].detach().item()
        else: 
            detached_loss[key] = loss_dict[key]
    detached_loss['epoch'] = iterations/num_iter_per_epoch
    wandb.log(detached_loss, step=iterations)








def train_func(config_path: str) -> None:
    """Train an experiment specified entirely by YAML config."""
    raw = load_config(config_path)
    config: ExperimentConfig = validate_config(raw)
    load_plugins(config.plugins)

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

    # Initialize the process group for distributed training
    # This initialization is needed to enable cross-GPU communication
    _dist_setup(enable_dist_train, device)

    rank = dist.get_rank() if enable_dist_train else 0
    world_size = dist.get_world_size() if enable_dist_train else 1

    if rank == 0:
        # Pass the config dictionary so you can filter by hyperparameters in the UI
        # You might need to add 'project_name' to your yaml schema, or hardcode it here
        project_name = config.data.datamodule.params["task_name"]
        
        wandb.init(
            project=project_name,
            config=_params_dict(config), # Uses your helper to dump Pydantic config
            name=f"{getattr(config.train, f'{project_name}', 'imitation_learning')}", # Optional: Readable run name
            # reinit=True # Uncomment if you run multiple trainings in one script execution
        )

    # Same seed on all ranks for synchronizing model initialization weights
    base_seed = getattr(config.train, "seed", 0)
    set_global_seed(seed=base_seed)

    if rank == 0: print(f"Global batch size = {config.data.batch_size * world_size}")

    trainer = _build_trainer(world_size, rank, local_rank, enable_dist_train, config, device)
    _dist_barrier(enable_dist_train, local_rank)

    # After models build with synchronized model initialization, offset seed for runtime randomness (Dropout, etc.)
    # This ensures distinct stochastic behavior per GPU
    set_global_seed(seed=base_seed + rank)
    _dist_barrier(enable_dist_train, local_rank)
    dataloader, sampler, stats = _build_dataloader(config=config, world_rank=rank, local_rank=local_rank, world_size=world_size, enable_dist_train=enable_dist_train)
    num_iter_per_epoch = float(len(dataloader))
    try:
        stats_cpu = tree_map(map_list_to_torch, stats)

        iterations = 0
        epoch = 0

        offline_iter = iter(dataloader)

        # Get handle to the global replay buffer
        replay_buffer = ray.get_actor("replay_buffer")
        
        # Get handle to policy state manager
        policy_state_manager = ray.get_actor("policy_state_manager")

        while True:
            # --- Source A: Offline Data ---
            try:
                offline_data = next(offline_iter)
            except StopIteration:
                epoch += 1
                dataloader.sampler.set_epoch(epoch)
                offline_iter = iter(dataloader) # Restart epoch
                offline_data = next(offline_iter)

            # --- Source B: Online Data ---
            # Each GPU asks the buffer for data independently.
            # Since the buffer is random, it's okay if they sample independently.
            # (Or you can shard this too, but random sampling is usually sufficient)
            future = replay_buffer.sample.remote(batch_size=config.data.batch_size)
            online_data = ray.get(future)

            # --- Combine & Train ---
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # normalize offline data
                offline_data['action'] = (offline_data['action'] - stats_cpu['action']['mean']) / (stats_cpu['action']['std'] + 1e-8)
                offline_data['observation.state'] = (offline_data['observation.state'] - stats_cpu['observation.state']['mean']) / (stats_cpu['observation.state']['std'] + 1e-8)
                offline_data['observation.current'] = (offline_data['observation.current'] - stats_cpu['observation.current']['mean']) / (stats_cpu['observation.current']['std'] + 1e-8)
                offline_data['observation.proprio_state'] = offline_data['observation.state']
                offline_data['observation.state'] = torch.concat([offline_data['observation.state'], offline_data['observation.current']], dim=-1)
                
                
                # combine online & offline data
                # online_data
                data = offline_data

                # normalize online data
                data = cast_dtype(data, torch.float32)
                loss_dict = trainer.train_step(data=move_to_device(data, device), iterations=iterations)
                if rank == 0:
                    _record(loss_dict, iterations, num_iter_per_epoch)
                    print(f"{iterations} iterations complete")
                    # Need to check inside save_checkpoints if the models are wrapped by DDP
                    if (iterations + 1) % config.train.save_every == 0:
                        _save_checkpoints(models=trainer.models, 
                                          optimizers=trainer.optimizers, 
                                          save_dir=config.train.save_dir, 
                                          epoch=0)
                        
                        # send the policy weight to the inference engine
                        policy_components_weights = {}
                        # move the state dict to CPU to use ray.put, which works with CPU shared memory
                        for model_name in trainer.models.keys():
                            policy_components_weights[model_name] = {k: v.cpu() for k, v in 
                                                                     trainer.models[model_name].state_dict().items()}
                        weights_ref = ray.put(policy_components_weights) # Push heavy data to Plasma
                        policy_state_manager.update_weights.remote(weights_ref) # Push light reference
            
            iterations += 1 # has to be updated for all workers
            gc.collect() 
            torch.cuda.empty_cache()

    finally:
        if rank == 0:
            print("program terminating...")
            wandb.finish()
