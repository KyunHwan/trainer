import torch
import torch.nn as nn
import einops

from flow_matching.path.affine import AffineProbPath
from flow_matching.path.scheduler.scheduler import CondOTScheduler

import torch.nn.functional as F

from offline_trainer.registry import TRAINER_REGISTRY
from typing import Any
import copy


@TRAINER_REGISTRY.register("mutual_information_estimator_trainer")
class Naive_Flow_Matching_Policy_Trainer(nn.Module):
    def __init__(self,
                 *,
                 models: nn.ModuleDict, 
                 optimizers: dict[str, torch.optim.Optimizer],
                 loss: nn.Module,
                 device):
        super().__init__()
        
        self.models=models
        self.optimizers=optimizers
        self.loss=loss

    def forward(self, data: dict[str, Any], epoch, total_epochs) -> dict[str, torch.Tensor]:
        loss = {}


        """ Action """
        encoded_action = self.models['action_encoder'](data['action'])['embedding']
        action_recon = self.models['action_decoder'](encoded_action + torch.randn_like(encoded_action))['action']

        """ State """
        with torch.no_grad():
            state_vae_input = {
                'cam_1': F.interpolate(
                            data['observation.images.cam_head'],
                            size=(128, 128),
                            mode="bilinear",
                            align_corners=False,),
                'cam_2': F.interpolate(
                            data['observation.images.cam_left'],
                            size=(128, 128),
                            mode="bilinear",
                            align_corners=False,),
                'cam_3': F.interpolate(
                            data['observation.images.cam_right'],
                            size=(128, 128),
                            mode="bilinear",
                            align_corners=False,),
                'state': data['observation.state']
            }

        encoded_state = self.models['state_encoder'](state_vae_input)['embedding']
        recon_output = self.models['state_decoder'](encoded_state + torch.randn_like(encoded_state))

        img_recon_loss = 0.0
        for key in state_vae_input.keys():
            img_recon_loss += (recon_output[key] - state_vae_input[key]).pow(2).mean(2)
        loss["total"] = 0.01 * ((encoded_action - torch.zeros_like(encoded_action)).pow(2).mean() +\
                                (encoded_state - torch.zeros_like(encoded_state)).pow(2).mean()) +\
                        0.005 * img_recon_loss
            
        return loss

    def train_step(self, data: dict[str, Any], epoch, total_epochs, iterations) -> dict[str, Any]:
        self._ready_train()
        self._zero_grad()

        loss = self.forward(data, epoch, total_epochs)

        self._backward(loss)
        self._step()
        
        detached_loss = self._detached_loss(loss)
        
        return detached_loss

    def _ready_train(self):
        for key in self.optimizers.keys():
            self.models[key].train()
            self.optimizers[key].train()
            
    def _zero_grad(self):
        for key in self.optimizers.keys():
            self.optimizers[key].zero_grad(set_to_none=True)

    def _backward(self, loss: dict[str, Any]):
        # can do backbward independently on each loss since they're from disjoint graphs
        for key in loss.keys():
            if isinstance(loss[key], torch.Tensor):
                loss[key].backward()

    def _step(self):
        for key in self.optimizers.keys():
            self.optimizers[key].step()
            
    def _detached_loss(self, loss: dict[str, Any]):
        detached_loss = {}
        for key in loss.keys():
            if isinstance(loss[key], torch.Tensor):
                detached_loss[key] = loss[key].detach().item()
            else:
                detached_loss[key] = loss[key]
        return detached_loss
