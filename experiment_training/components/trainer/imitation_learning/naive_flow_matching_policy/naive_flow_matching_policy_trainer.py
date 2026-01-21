import torch
import torch.nn as nn
import einops

from flow_matching.path.affine import AffineProbPath
from flow_matching.path.scheduler.scheduler import CondOTScheduler

import torch.nn.functional as F

from trainer.registry import TRAINER_REGISTRY
from typing import Any
import copy


@TRAINER_REGISTRY.register("naive_flow_matching_policy_trainer")
class Naive_Flow_Matching_Policy_Trainer(nn.Module):
    def __init__(self,
                 *,
                 models: nn.ModuleDict, 
                 optimizers: dict[str, torch.optim.Optimizer],
                 loss: nn.Module,
                 device):
        super().__init__()
        
        self.dist = torch.distributions.Beta(1.0, 1.5) # as in pi0-paper by physical intelligence
        self.probPath = AffineProbPath(CondOTScheduler())
        
        self.models=models
        self.optimizers=optimizers
        self.loss=loss

    def forward(self, data: dict[str, Any], epoch, total_epochs) -> dict[str, torch.Tensor]:
        loss = {}

        """ Backbone """
        with torch.no_grad():
            """ Radiov3 """
            head_image_features, head_image_semantic = self.models['backbone'](data['observation.images.cam_head'])
            head_image_features = einops.rearrange(head_image_features, 'b c h w -> b 1 c h w')

            left_image_features, _ = self.models['backbone'](data['observation.images.cam_left'])
            left_image_features = einops.rearrange(left_image_features, 'b c h w -> b 1 c h w')

            right_image_features, _ = self.models['backbone'](data['observation.images.cam_right'])
            right_image_features = einops.rearrange(right_image_features, 'b c h w -> b 1 c h w')

            """ Depth """
            # outputs (batch, num_features, height, width, feature_dim) shaped latent features
            depth_head = self.models['da3'](image=data['observation.images.cam_head'], export_feat_layers=[8, 13, 18, 23])
            depth_left = self.models['da3'](image=data['observation.images.cam_left'], export_feat_layers=[8, 13, 18, 23])
            depth_right = self.models['da3'](image=data['observation.images.cam_right'], export_feat_layers=[8, 13, 18, 23])

        """ Proprio Projection """
        # Assumes that proprio feature dimension will be matched to that of visual
        conditioning_info = self.models['proprio_projector'](cond_proprio=data['observation.state'],
                                                            cond_visual=torch.cat([head_image_features,
                                                                                    left_image_features, 
                                                                                    right_image_features],
                                                                                    dim=1),)

        """ Flow Matching """
        noise = torch.randn_like(data['action'], device=data['action'].device)
        time = self.dist.sample((data['action'].shape[0],)).to(data['action'].device)
        sample = self.probPath.sample(t=time, x_0=noise, x_1=data['action'])

        x_t = sample.x_t
        dx_t = sample.dx_t

        dx_t_hat = self.models['action_decoder'](time=time, 
                                                 noise=x_t, 
                                                 memory_input=torch.cat([einops.rearrange(depth_head, 'b n h w d -> b (n h w) d'),
                                                                         einops.rearrange(depth_left, 'b n h w d -> b (n h w) d'),
                                                                         einops.rearrange(depth_right, 'b n h w d -> b (n h w) d'),
                                                                         conditioning_info], dim=1),
                                                 discrete_semantic_input=head_image_semantic,)
        
        err = (dx_t - dx_t_hat).pow(2)
        velocity_loss = err.mean()
        
        loss["velocity"] = velocity_loss
            
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
            if hasattr(self.optimizers[key], 'train'): 
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
