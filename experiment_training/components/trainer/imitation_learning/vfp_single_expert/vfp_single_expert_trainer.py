import torch
import torch.nn as nn
import einops

from flow_matching.path.affine import AffineProbPath
from flow_matching.path.scheduler.scheduler import CondOTScheduler

import torch.nn.functional as F

from trainer.trainer.registry import TRAINER_REGISTRY
from typing import Any
import copy

import torch
import torch.distributed as distributed

import math
import random

def router_z_loss(logits: torch.Tensor) -> torch.Tensor:
    # ST-MoE style: mean(logsumexp(logits)^2)
    return torch.mean(torch.logsumexp(logits, dim=-1) ** 2)


def switch_load_balancing_loss(
    router_probs: torch.Tensor,  # [B, E]
    routing_map: torch.Tensor,   # [B, E] 0/1
    top_k: int
) -> torch.Tensor:
    """
    Switch-style auxiliary loss:
      LB = E * sum_i f_i * P_i
    where:
      f_i = fraction of tokens routed to expert i (counts over B*top_k routes)
      P_i = average router probability mass to expert i
    """
    B, E = router_probs.shape
    f = routing_map.sum(0) / (B * top_k)
    P = router_probs.mean(0)
    return E * torch.sum(f * P)

@TRAINER_REGISTRY.register("vfp_single_expert_trainer")
class VFP_Single_Expert_Trainer(nn.Module):
    def __init__(self,
                 *,
                 models: nn.ModuleDict, 
                 optimizers: dict[str, torch.optim.Optimizer],
                 loss: nn.Module,
                 device):
        super().__init__()
        
        self.dist = torch.distributions.Beta(1.0, 1.5) # as in pi0-paper by physical intelligence
        self.probPath = AffineProbPath(CondOTScheduler())
        self.device = device
        
        self.models=models
        self.optimizers=optimizers
        self.loss=loss

    def forward(self, data: dict[str, Any], epoch, total_epochs, iterations) -> dict[str, torch.Tensor]:
        loss = {}

        """ Backbone """
        # head_image_features, head_image_semantic = self.models['backbone'](data['observation.images.cam_head'])
        # head_image_features = einops.rearrange(head_image_features, 'b c h w -> b 1 c h w')

        # left_image_features, _ = self.models['backbone'](data['observation.images.cam_left'])
        # left_image_features = einops.rearrange(left_image_features, 'b c h w -> b 1 c h w')

        # right_image_features, _ = self.models['backbone'](data['observation.images.cam_right'])
        # right_image_features = einops.rearrange(right_image_features, 'b c h w -> b 1 c h w')
        batch_size = data['observation.images.cam_head'].shape[0]

        cam_names = ['observation.images.cam_head', 'observation.images.cam_left', 'observation.images.cam_right']
        selected_cam = random.choice(cam_names)
        if random.random() < 0.15:
            data[selected_cam] = torch.zeros_like(data[selected_cam])

        head_image_features, head_image_semantic = self.models['head_backbone'](data['observation.images.cam_head'])
        left_image_features, left_image_semantic = self.models['left_backbone'](data['observation.images.cam_left'])
        right_image_features, right_image_semantic = self.models['right_backbone'](data['observation.images.cam_right'])

        head_image_features = einops.rearrange(head_image_features, 'b c h w -> b (h w) c')
        head_image_semantic = einops.rearrange(head_image_semantic, 'b d -> b 1 d')
        
        left_image_features = einops.rearrange(left_image_features, 'b c h w -> b (h w) c')
        left_image_semantic = einops.rearrange(left_image_semantic, 'b d -> b 1 d')
        
        right_image_features = einops.rearrange(right_image_features, 'b c h w -> b (h w) c')
        right_image_semantic = einops.rearrange(right_image_semantic, 'b d -> b 1 d')

        # with torch.no_grad():
        #     """ Depth """
        #     # outputs (batch, num_features, height, width, feature_dim) shaped latent features
        #     depth_head = einops.rearrange(self.models['da3'](image=data['observation.images.cam_head'], export_feat_layers=[18, 12]),
        #                                   'b n h w d -> b (n h w) d')
        #     # outputs (batch, num_features, height, width, feature_dim) shaped latent features
        #     depth_left = einops.rearrange(self.models['da3'](image=data['observation.images.cam_left'], export_feat_layers=[18, 23]),
        #                                   'b n h w d -> b (n h w) d')
        #     # outputs (batch, num_features, height, width, feature_dim) shaped latent features
        #     depth_right = einops.rearrange(self.models['da3'](image=data['observation.images.cam_right'], export_feat_layers=[18, 23]),
        #                                   'b n h w d -> b (n h w) d')

        # """ VQVAE Posterior """
        # posterior_cls_token = self.models['vae_posterior'](cond_proprio=data['observation.state'],
        #                                                      cond_visual=head_image_features,
        #                                                      cond_semantic=head_image_semantic,
        #                                                      action = data['action']
        #                                                      )
        
        # """ VQVAE Prior """
        # prior_cls_token = self.models['vae_prior'](cond_proprio=data['observation.state'],
        #                                              cond_visual=head_image_features,
        #                                              cond_semantic=head_image_semantic,
        #                                              action = None
        #                                              )
        # head_image_features = einops.rearrange(head_image_features, 'b n c h w -> b (n h w) c')
        # kl = torch.pow(prior_cls_token - posterior_cls_token, 2).mean()

        """ Proprio Projection """
        conditioning_info = self.models['info_embedder'](
            cond_proprio=data['observation.state'],
            cond_visual=[
                head_image_features,
                #depth_head,
                left_image_features, # einops.rearrange(left_image_features, 'b n c h w -> b (n h w) c'),
                #depth_left,
                right_image_features, #einops.rearrange(right_image_features, 'b n c h w -> b (n h w) c')
                #depth_right
            ],
            cond_semantic=[
                head_image_semantic,
                left_image_semantic,
                right_image_semantic
            ],
            action=None
        )['encoder_output']

        """ Flow Matching """
        noise = torch.randn_like(data['action'], device=data['action'].device)
        time = self.dist.sample((data['action'].shape[0],)).to(data['action'].device)
        sample = self.probPath.sample(t=time, x_0=noise, x_1=data['action'])

        x_t = sample.x_t
        dx_t = sample.dx_t

        B = data['action'].shape[0]

        # concat_actions = self.models['action_decoder'](
        #                             time=torch.cat([time, torch.zeros_like(time)], dim=0), 
        #                             noise=torch.cat([x_t, noise], dim=0), 
        #                             memory_input=torch.cat([conditioning_info, conditioning_info], dim=0),
        #                             discrete_semantic_input=None,
        #                         ) # (2 * b s d)
        # velocity_loss = (dx_t - concat_actions[:B]).pow(2).mean()

        # sinkhorn_loss = self.loss(pred_action = noise + concat_actions[B:], 
        #                           target_action = data['action'], 
        #                           state_pred = data['observation.state'], 
        #                           state_target = data['observation.state'])
        actions = self.models['action_decoder'](
            time=time,
            noise=x_t,
            memory_input=conditioning_info,
            discrete_semantic_input=None
        )
        velocity_loss = (dx_t - actions).pow(2).mean()
        loss["total"] = velocity_loss #+ 0.2 * sinkhorn_loss# + kl
        #loss["prior_posterior"] = kl.detach().clone().item()
        loss["velocity"] = velocity_loss.detach().clone().item()
        #loss["Sinkhorn"] = sinkhorn_loss.detach().clone().item()
        
        return loss 




    def train_step(self, data: dict[str, Any], epoch, total_epochs, iterations) -> dict[str, Any]:
        self._ready_train()
        self._zero_grad()

        loss = self.forward(data, epoch, total_epochs, iterations)

        self._backward(loss)
        detached_loss = self._clip_get_grad_norm(loss=loss, clip_val=1.0)
        self._step()
        detached_loss = self._detached_loss(detached_loss)
        detached_loss = self._get_lr(detached_loss)
        return detached_loss
    
    def unwrap(self, m: nn.Module) -> nn.Module:
        # DDP / DataParallel expose `.module`; plain models don't.
        return m.module if hasattr(m, "module") else m

    @torch.no_grad()
    def update_posterior_ema(self):
        src = self.unwrap(self.models["vqvae_posterior"])
        msd = src.state_dict()
        for k, v_ema in self.posterior_ema.state_dict().items():
            v = msd[k]
            if v.dtype.is_floating_point:
                v_ema.mul_(self.posterior_ema_factor).add_(v, alpha=1.0 - self.posterior_ema_factor)
            else:
                v_ema.copy_(v)

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
    
    def _clip_get_grad_norm(self, loss, clip_val: float=float('inf')):
        for model_name in self.models.keys():
            if model_name in self.optimizers.keys():
                loss[f"{model_name} grad_norm"] = torch.nn.utils.clip_grad_norm_(self.models[model_name].parameters(), max_norm=clip_val).detach().item()
        return loss
    
    def _get_lr(self, loss):
        for model_name in self.models.keys():
            if model_name in self.optimizers.keys():
                loss[f"{model_name} lr"] = self.optimizers[model_name].param_groups[0]['lr']
        return loss
