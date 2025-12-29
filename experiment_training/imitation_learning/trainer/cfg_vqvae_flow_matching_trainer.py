import torch
import torch.nn as nn
import einops

from flow_matching.path.affine import AffineProbPath
from flow_matching.path.scheduler.scheduler import CondOTScheduler

import torch.nn.functional as F

from offline_trainer.registry import TRAINER_REGISTRY
from typing import Any


@TRAINER_REGISTRY.register("cfg_vqvae_flow_matching_trainer")
class CFG_VQVAE_Flow_Matching_Trainer(nn.Module):
    def __init__(self,
                 *,
                 models: nn.ModuleDict, 
                 optimizers: dict[str, torch.optim.Optimizer],
                 loss: nn.Module,):
        super().__init__()
        
        self.dist = torch.distributions.Beta(1.0, 1.5) # as in pi-paper by physical intelligence
        self.probPath = AffineProbPath(CondOTScheduler())
        
        self.models=models
        self.optimizers=optimizers
        self.loss=loss

    def forward(self, data: dict[str, Any]) -> dict[str, torch.Tensor]:
        loss = {}

        """ Backbone """
        with torch.no_grad():
            head_image_features, head_image_semantic = self.models['backbone'](data['images']['head'])
            head_image_features = einops.rearrange(head_image_features, 'b c h w -> b 1 c h w')

            left_image_features, _ = self.models['backbone'](data['images']['left'])
            left_image_features = einops.rearrange(left_image_features, 'b c h w -> b 1 c h w')

            right_image_features, _ = self.models['backbone'](data['images']['right'])
            right_image_features = einops.rearrange(right_image_features, 'b c h w -> b 1 c h w')


        """ VQVAE Posterior """
        posterior_cls_token = self.models['vqvae_posterior'](cond_proprio=data['proprio'],
                                                             cond_visual=head_image_features,
                                                             cond_semantic=head_image_semantic,
                                                             action = data['action']
                                                             )

        """ VQVAE Prior """

        prior_cls_token = self.models['vqvae_prior'](cond_proprio=data['proprio'],
                                                     cond_visual=head_image_features,
                                                     cond_semantic=head_image_semantic,
                                                     action = None
                                                     )

        """ VQVAE Codebook """
        related_codebook_quantized_vec = self.models['vqvae_codebook'](continuous_vec=posterior_cls_token)

        """ NSVQ """
        # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9696322
        noise_vec = torch.randn_like(posterior_cls_token, device=posterior_cls_token.device)
        normalized_noise_vec = F.normalize(noise_vec, p=2, dim=0)
        simulated_quantized_vec = posterior_cls_token + \
                                  normalized_noise_vec * torch.norm(posterior_cls_token - related_codebook_quantized_vec, p=2, dim=0)

        """ Info Encoder """
        conditioning_info = self.models['info_encoder'](cond_proprio=data['proprio'],
                                                        cond_visual=torch.cat([head_image_features, 
                                                                               left_image_features, 
                                                                               right_image_features],
                                                                               axis=1),
                                                        cond_semantic=simulated_quantized_vec)

        """ Flow Matching """
        noise = torch.randn_like(data['action'], device=data['action'].device)
        time = self.dist.sample((data['action'].shape[0],)).to(data['action'].device)
        sample = self.probPath.sample(t=time, x_0=noise, x_1=data['action'])

        x_t = sample.x_t
        dx_t = sample.dx_t

        dx_t_hat = self.models['action_decoder'](time=time, 
                                                 noise=x_t, 
                                                 memory_input=conditioning_info, 
                                                 discrete_semantic_input=simulated_quantized_vec,)
        
        min_T = min(data['action'].shape[1], dx_t_hat.shape[1])
        dx_t_hat = dx_t_hat[:, :min_T]
        dx_t = dx_t[:, :min_T]
        is_pad = data['is_pad'][:, :min_T]
        
        velocity_loss = self.loss(dx_t_hat, dx_t)

        valid_mask = (~is_pad).unsqueeze(-1)
        masked_velocity_loss = velocity_loss * valid_mask

        loss["velocity"] = masked_velocity_loss.mean()
        loss["prior_posterior"] = self.loss(prior_cls_token, posterior_cls_token.detach()).mean()

        return loss


    def train_step(self, data: dict[str, Any]) -> dict[str, Any]:
        self._ready_train()
        self._zero_grad()

        loss = self.forward(data)

        self._backward(loss)
        self._step()
        
        return loss
    
    def _ready_train(self):
        for key in self.models.keys():
            if key != 'backbone': self.models[key].train()
            else: self.models[key].eval()
            self.optimizers[key].train()

    def _zero_grad(self):
        for key in self.optimizers.keys():
            self.optimizers[key].zero_grad()

    def _backward(self, loss: dict[str, torch.Tensor]):
        # can do backbward independently on each loss since they're from disjoint graphs
        for key in loss.keys():
            loss[key].backward()

    def _step(self):
        for key in self.optimizers.keys():
            self.optimizers[key].step()

    def _detached_loss(self, loss: dict[str, torch.Tensor]):
        detached_loss = {}
        for key in loss.keys():
            detached_loss[key] = loss[key].detach().cpu().item()
        return detached_loss
