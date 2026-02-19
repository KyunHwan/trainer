"""Public API entrypoint for training."""
from __future__ import annotations

import os
import gc
from pathlib import Path

import torch
torch.autograd.set_detect_anomaly(True)

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

def _dist_setup(enable_dist_train, device) -> None:
    if enable_dist_train:
        dist.init_process_group(backend="nccl",
                                init_method="env://",
                                )

def _dist_barrier(dist_enabled, local_rank) -> None:
    if dist_enabled:
        dist.barrier()#device_ids=[local_rank])

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
        if local_rank == 0: 
            n_params_m = sum(p.numel() for p in policy.parameters()) / 1000000.0
            model_total_params += n_params_m
            print(f"Parameters of {k} model: {n_params_m:.1f} M")

        # Load model checkpoints / initialization
        # Need to check that saved models weren't wrapped using DDP (ie. that they aren't wrapped using modules)
        if config.train.load_dir is not None:
            path = os.path.join(config.train.load_dir, f"{k}.pt")
            if os.path.isfile(path):
                policy.load_state_dict(torch.load(path, map_location='cpu'))
            else:
                print(f"{path} doesn't exist as a file!")
        else:
            if config.model.component_build_args[k]['init']: 
                policy.apply(init_weights)
        
        if config.model.component_build_args[k]['freeze']:
            for param in policy.parameters():
                param.requires_grad_(False)
            policy.eval()
            policy = policy.to(device)
            models[k] = policy
            continue

        # If BatchNorm layers in a real model, convert to SyncBatchNorm so BN stats sync across replicas
        # This should be before moving the model onto a device
        if torch.cuda.is_available() and enable_dist_train and any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in policy.modules()):
            policy = nn.SyncBatchNorm.convert_sync_batchnorm(policy)

        # If the model wasn't loaded onto device, load it onto device
        policy = policy.to(device)
        
        # 'find_unused_parameters' is used when there are conditional cases that leave some parts of the model unexplored. 
        # For example, when mixture of experts is used.
        find_unused_parameters = getattr(config.model, "find_unused_parameters", False)

        models[k] = DDP(policy, find_unused_parameters=find_unused_parameters, device_ids=[local_rank], output_device=local_rank)

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
            path = os.path.join(config.train.load_dir, f"{model_name}_opt.pt")
            if os.path.isfile(path):
                optimizers[model_name].load_state_dict(torch.load(path, map_location=device))
            else:
                print(f"{path} doesn't exist as a file!")
    
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
                    # Construct the full path
                    save_dir = config.train.save_dir
                    stats_path = os.path.join(save_dir, "dataset_stats.pkl")
                    
                    # 1. Expand user if necessary (e.g. handle '~')
                    stats_path = os.path.expanduser(stats_path)
                    dir_name = os.path.dirname(stats_path)

                    # 2. Create the directory if it doesn't exist
                    os.makedirs(dir_name, exist_ok=True)

                    # 3. Now safe to write the file
                    with open(stats_path, "wb") as f:
                        pickle.dump(stats, f)

    else: 
        dataset = returned_product
        stats = None

    # When using DistributedSampler, do NOT set shuffle=True on DataLoader.
    # Shuffling is handled by the sampler (see PyTorch DDP tutorial pattern)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=world_rank, drop_last=True) if enable_dist_train else RandomSampler(dataset)
    dataloader = DataLoader(dataset, 
                            sampler=sampler, 
                            batch_size=config.data.batch_size,
                            num_workers=config.data.num_workers,
                            pin_memory=config.data.pin_memory,
                            persistent_workers=config.data.persistent_workers,
                            prefetch_factor=config.data.prefetch_factor,
                            worker_init_fn=seed_worker,
                            drop_last=False,
                            shuffle=False)
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








def train(config_path: str) -> None:
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
            name=config.train.project_name,
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
    _dist_barrier(enable_dist_train, local_rank)
    num_iter_per_epoch = float(len(dataloader))
    try:
        stats_cpu = tree_map(map_list_to_torch, stats)

        iterations = 0
        
        for epoch in range(config.train.epoch):
            if enable_dist_train:
                sampler.set_epoch(epoch)
        
            for _, data in enumerate(tqdm(dataloader, disable=(rank != 0))):
                #print(stats_cpu['action']['mean'].shape) # (24)
                data['action'] = (data['action'] - stats_cpu['action']['mean']) / (stats_cpu['action']['std'] + 1e-8)
                data['observation.state'] = (data['observation.state'] - stats_cpu['observation.state']['mean']) / (stats_cpu['observation.state']['std'] + 1e-8)
                data['observation.current'] = (data['observation.current'] - stats_cpu['observation.current']['mean']) / (stats_cpu['observation.current']['std'] + 1e-8)
                data['observation.proprio_state'] = data['observation.state']
                data['observation.state'] = torch.concat([data['observation.state'], data['observation.current']], dim=-1)
                data = cast_dtype(data, torch.float32)
                data = move_to_device(data, device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss_dict = trainer.train_step(data=data, epoch=epoch, total_epochs=config.train.epoch, iterations=iterations)
                if rank == 0:
                    _record(loss_dict, iterations, num_iter_per_epoch)
                iterations += 1 # has to be updated for all GPUs
            # _dist_barrier(enable_dist_train, local_rank)

            if rank == 0:
                print(f"Epoch {epoch} complete")
                # Need to check inside save_checkpoints if the models are wrapped by DDP
                if (epoch + 1) % config.train.save_every == 0:
                    _save_checkpoints(models=trainer.models, 
                                    optimizers=trainer.optimizers, 
                                    save_dir=config.train.save_dir, 
                                    epoch=epoch + 1)
            gc.collect() 
            torch.cuda.empty_cache()
            _dist_barrier(enable_dist_train, local_rank)

            # gc.collect() 
            # torch.cuda.empty_cache()
            # _dist_barrier(enable_dist_train, local_rank)
            
        if rank == 0: 
            print("Training finished !!")

    finally:
        if rank == 0:
            print("program terminating...")
            wandb.finish()
        _dist_cleanup(enable_dist_train)









if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    parser = argparse.ArgumentParser(description="Parse for train config .yaml file")
    parser.add_argument("--train_config", help="absolute path to the train config .yaml file.", required=True)
    args = parser.parse_args()
    #test()
    #ddp_broadcast_test()
    train(args.train_config)
