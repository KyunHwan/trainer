"""Public API entrypoint for training."""
from __future__ import annotations

import os

from offline_trainer.config.loader import load_config
from offline_trainer.config.schemas import ExperimentConfig, validate_config
from offline_trainer.modeling.factories import PolicyConstructorModelFactory
from offline_trainer.registry import (
    TRAINER_REGISTRY,
    DATASET_BUILDER_REGISTRY,
    OPTIMIZER_BUILDER_REGISTRY,
    LOSS_BUILDER_REGISTRY,
)

from offline_trainer.templates import (
    DatasetFactory,
    LossFactory,
    OptimizerFactory,
    Trainer
)
from offline_trainer.registry.plugins import load_plugins
from offline_trainer.utils.import_utils import instantiate
from offline_trainer.utils.seed import *
import argparse

from offline_trainer.utils.device import move_to_device

import torch
import torch.nn as nn
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler

from tqdm import tqdm

def _params_dict(params) -> dict:
    if hasattr(params, "model_dump"):
        return params.model_dump()
    return params

""" Distributed Training Helpers """

def _dist_setup(enable_dist_train) -> None:
    if enable_dist_train:
        dist.init_process_group(backend="nccl")

def _dist_barrier(dist_enabled) -> None:
    if dist_enabled:
        dist.barrier()

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
    return loss_fn if not isinstance(loss_fn, nn.Module) else loss_fn.to(device)

def _build_models(local_rank, enable_dist_train, config, config_path, device) -> nn.ModuleDict[str, nn.Module]:
    """
    Required Config Params:
        model.config_path
        train.load_dir
    """
    if config.model.config_path and not os.path.isabs(config.model.config_path):
        config.model.config_path = os.path.abspath(
            os.path.join(os.path.dirname(config_path), config.model.config_path)
        )

    model_factory = PolicyConstructorModelFactory()
    model = model_factory.build(config.model.as_dict())

    # Allows for multiple models to be trained (ex. in offline rl setting)
    models = {"main": model} if not isinstance(model, dict) else model

    for k, policy in models.items():
        print(f"Total parameters of {k} model: {sum(p.numel() for p in policy.parameters())}")
        
        # Load model checkpoints
        # Need to check that saved models weren't wrapped using DDP (ie. that they aren't wrapped using modules)
        if config.train.load_dir is not None:
            policy.load_state_dict(torch.load(os.path.join(config.train.load_dir, f"{k}.pt"), map_location=device))

        # If BatchNorm layers in a real model, convert to SyncBatchNorm so BN stats sync across replicas
        if torch.cuda.is_available() and enable_dist_train and any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in policy.modules()):
            policy = nn.SyncBatchNorm.convert_sync_batchnorm(policy)

        policy = policy.to(device)
        
        # 'find_unused_parameters' is used when there are conditional cases that leave some parts of the model unexplored. 
        # For example, when mixture of experts is used.
        find_unused_parameters = getattr(config.model, "find_unused_parameters", False)
        models[k] = DDP(policy, device_ids=[local_rank], find_unused_parameters=find_unused_parameters) if enable_dist_train else policy

    return nn.ModuleDict(models)

def _build_optimizers(config, models: nn.ModuleDict[str, nn.Module], device) -> dict[str, torch.optim.Optimizer]:
    """
    Required Config Params:
        train.optimizer.type
        train.optimizer.params
        train.load_dir
    """
    optimizer_cls = OPTIMIZER_BUILDER_REGISTRY.get(config.train.optimizer.type)
    optimizer_factory = instantiate(optimizer_cls, _params_dict(config.train.optimizer.params))
    
    # One optimizer per model
    optimizers = {}
    for k, model in models.items():
        optimizers[k] = optimizer_factory.build(model.parameters())
        if config.train.load_dir is not None:
            optimizers[k].load_state_dict(torch.load(os.path.join(config.train.load_dir, f"{k}_optimizer.pt"), map_location=device))
    
    return optimizers

def _build_dataloader(rank, world_size, config, enable_dist_train):
    """
    Required Config Params:
        data.datamodule.type
        data.datamodule.params
        data.batch_size
        data.num_workers
        data.pin_memory
        data.persistent_workers 
    """
    
    datamodule_cls = DATASET_BUILDER_REGISTRY.get(config.data.datamodule.type)
    dataset = instantiate(datamodule_cls, config.data.datamodule.params, config=config)

    # When using DistributedSampler, do NOT set shuffle=True on DataLoader.
    # Shuffling is handled by the sampler (see PyTorch DDP tutorial pattern)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, drop_last=True) if enable_dist_train else RandomSampler(dataset)
    dataloader = DataLoader(dataset, 
                            sampler=sampler, 
                            batch_size=config.data.batch_size,
                            num_workers=config.data.num_workers,
                            pin_memory=config.data.pin_memory,
                            persistent_workers=config.data.persistent_workers,
                            worker_init_fn=seed_worker,
                            drop_last=False,
                            shuffle=False)
    return dataloader, sampler

def _build_trainer(local_rank, enable_dist_train, config, config_path, device) -> Trainer:
    """
    Required Config Params:
        train.trainer.type
        train.trainer.params
    """
    models = _build_models(local_rank, enable_dist_train, config, config_path, device)
    optimizers = _build_optimizers(config, models, device)
    loss_fn = _build_loss(config, device)

    trainer_cls = TRAINER_REGISTRY.get(config.train.trainer.type)
    trainer = instantiate(
        trainer_cls,
        _params_dict(config.train.trainer.params),
        models=models,
        optimizers=optimizers,
        loss_fn=loss_fn 
    )
    if not isinstance(trainer, Trainer):
        raise TypeError("Constructed object does not match Trainer interface")
    return trainer


""" Parameter Saving """

def _save_checkpoints(models: nn.ModuleDict[str, nn.Module], 
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
        model_filename = f"{key}_{epoch}.pt"
        model_path = os.path.join(epoch_folder, model_filename)

        # Check if wrapped
        # Access the original model to save "clean" weights
        state_to_save = models[key].module.state_dict() if isinstance(models[key], DDP) else models[key].state_dict()

        torch.save(state_to_save, model_path)
        
        # --- Save Optimizer ---
        # We assume the key exists in optimizers as stated in the prompt
        if key in optimizers:
            opt_filename = f"{key}_opt_{epoch}.pt"
            opt_path = os.path.join(epoch_folder, opt_filename)
            
            torch.save(optimizers[key].state_dict(), opt_path)
            
    print(f"Saved checkpoints for epoch {epoch} at {epoch_folder}")








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
    _dist_setup(enable_dist_train)

    rank = dist.get_rank() if enable_dist_train else 0
    world_size = dist.get_world_size() if enable_dist_train else 1

    # Same seed on all ranks for synchronizing model initialization weights
    base_seed = getattr(config.train, "seed", 0)
    set_global_seed(seed=base_seed)

    print(f"Global batch size = {config.data.batch_size * world_size}")

    trainer = _build_trainer(local_rank, enable_dist_train, config, config_path, device)

    # After model build with synchronized model initialization, offset seed for runtime randomness (Dropout, etc.)
    # This ensures distinct stochastic behavior per GPU
    set_global_seed(seed=base_seed + rank)
    dataloader, sampler = _build_dataloader(rank, world_size, config, enable_dist_train)


    try:
        for epoch in range(config.train.epochs):
            if enable_dist_train:
                sampler.set_epoch(epoch)

            for _, data in enumerate(tqdm(dataloader, disable=(rank != 0))):
                loss_dict = trainer.train_step(data=move_to_device(data, device))

            _dist_barrier(enable_dist_train)

            if rank == 0:
                print(f"Epoch {epoch} complete")
                # Need to check inside save_checkpoints if the models are wrapped by DDP
                _save_checkpoints(models=trainer.models, 
                                  optimizers=trainer.optimizers, 
                                  save_dir=config.train.save_dir, 
                                  epoch=epoch)
                
            _dist_barrier(enable_dist_train)

        print("Training finished !!")

    finally:
        _dist_cleanup(enable_dist_train)

    









if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    parser = argparse.ArgumentParser(description="Parse for train config .yaml file")
    parser.add_argument("--train_config", help="absolute path to the train config .yaml file.", required=True)
    args = parser.parse_args()

    train(args.train_config)
