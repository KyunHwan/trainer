import torch
import torch.nn as nn
import einops

from flow_matching.path.affine import AffineProbPath
from flow_matching.path.scheduler.scheduler import CondOTScheduler

import torch.nn.functional as F

from offline_trainer.registry import TRAINER_REGISTRY
from typing import Any
import copy

import torch
import torch.distributed as distributed

import math


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

@TRAINER_REGISTRY.register("variational_flow_matching_policy_trainer")
class Variational_Flow_Matching_Policy_Trainer(nn.Module):
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
        self.posterior_ema = copy.deepcopy(self.unwrap(self.models["vqvae_posterior"])).to(device).eval()
        for p in self.posterior_ema.parameters():
            p.requires_grad_(False)
        self.posterior_ema_factor = 0.999

        self.velocity_loss_ema = 0.0
        self.velocity_loss_ema_factor = 0.995

    def forward(self, data: dict[str, Any], epoch, total_epochs, iterations) -> dict[str, torch.Tensor]:
        loss = {}

        """ Backbone """
        with torch.no_grad():
            head_image_features, head_image_semantic = self.models['backbone'](data['observation.images.cam_head'])
            head_image_features = einops.rearrange(head_image_features, 'b c h w -> b 1 c h w')

            left_image_features, _ = self.models['backbone'](data['observation.images.cam_left'])
            left_image_features = einops.rearrange(left_image_features, 'b c h w -> b 1 c h w')

            right_image_features, _ = self.models['backbone'](data['observation.images.cam_right'])
            right_image_features = einops.rearrange(right_image_features, 'b c h w -> b 1 c h w')

            """ Depth """
            # outputs (batch, num_features, height, width, feature_dim) shaped latent features
            depth_head = self.models['da3'](image=data['observation.images.cam_head'], export_feat_layers=[8, 13, 18, 23])

        """ VQVAE Posterior """
        posterior_cls_token = self.models['vqvae_posterior'](cond_proprio=data['observation.state'],
                                                             cond_visual=head_image_features,
                                                             cond_semantic=head_image_semantic,
                                                             action = data['action']
                                                             )

        """ VQVAE Prior """
        prior_cls_token = self.models['vqvae_prior'](cond_proprio=data['observation.state'],
                                                     cond_visual=head_image_features,
                                                     cond_semantic=head_image_semantic,
                                                     action = None
                                                     )
        
        kl = torch.pow(prior_cls_token - posterior_cls_token, 2).squeeze().sum(dim=1).mean()

        """ Proprio Projection """
        # Assumes that proprio feature dimension will be matched to that of visual
        conditioning_info = self.models['proprio_projector'](cond_proprio=data['observation.state'],
                                                            cond_visual=torch.cat([head_image_features,
                                                                                    left_image_features, 
                                                                                    right_image_features],
                                                                                    dim=1),)

        """ Gating """
        #gating = self.models['gate'](posterior_cls_token)
        gating = self.models['gate'](posterior_cls_token + torch.randn_like(posterior_cls_token), iterations=iterations, training=True, use_noise=True)
        B, E = gating.shape

        # expert_ids = gating.argmax(dim=1)          # (B,) each element in [0, E-1]

        """ Flow Matching """
        noise = torch.randn_like(data['action'], device=data['action'].device)
        time = self.dist.sample((data['action'].shape[0],)).to(data['action'].device)
        sample = self.probPath.sample(t=time, x_0=noise, x_1=data['action'])

        x_t = sample.x_t
        dx_t = sample.dx_t


        concat_moe_actions = self.models['moe_action_decoder'](
                                                time=torch.cat([time, torch.zeros_like(time)], dim=0), 
                                                noise=torch.cat([x_t, noise], dim=0), 
                                                memory_input=torch.cat([torch.cat([einops.rearrange(depth_head, 'b n h w d -> b (n h w) d'),
                                                                        conditioning_info], dim=1),
                                                                        torch.cat([einops.rearrange(depth_head, 'b n h w d -> b (n h w) d'),
                                                                        conditioning_info], dim=1)], dim=0),
                                                discrete_semantic_input=None,)
        concat_dx_t = einops.rearrange(torch.stack([dx_t for i in range(E)], dim=0), 'e b s d -> b e s d')
        err = einops.rearrange((concat_dx_t - concat_moe_actions[:B]).pow(2), 'b e s d -> b e (s d)')#.sum(dim=-1)
        moe_loss = (err * gating[:, :, None]).sum(dim=1).mean()

        averaged_moe_actions = torch.sum(concat_moe_actions[B:] * gating[:, :, None, None], dim=1)
        sinkhorn_loss = self.loss(pred_action = noise + averaged_moe_actions, 
                                  target_action = data['action'], 
                                  state_pred = data['observation.state'], 
                                  state_target = data['observation.state'])

        """ EMA """
        # # For prior training --> since posterior is a moving target, naively doing l2 distance between the two destabilizes prior learning 
        # with torch.no_grad():
        #     self.posterior_ema.eval()
        #     posterior_target = self.posterior_ema(cond_proprio=data['observation.state'],
        #                                           cond_visual=head_image_features,
        #                                           cond_semantic=head_image_semantic,
        #                                           action = data['action']
        #                                          )  # e.g., returns posterior_cls_token_ema
        #     loss["True_prior_posterior"] = torch.pow(prior_cls_token.detach().clone() - posterior_cls_token.detach().clone(), 2).mean().item()
        
        """ Activated Expert """
        with torch.no_grad():
            expert_ids = gating.detach().argmax(dim=1)  # (B_local,)

            # local histogram (E,)
            local_counts = torch.bincount(expert_ids, minlength=E).to(dtype=torch.long)

            # aggregate across ranks
            if distributed.is_available() and distributed.is_initialized():
                distributed.all_reduce(local_counts, op=distributed.ReduceOp.SUM)
            
            winner_id = int(local_counts.argmax().item())  # tie-break: lowest id wins

            total_samples = local_counts.sum()
            probs = local_counts / total_samples  # Shape: (E,)
            indices = torch.arange(len(local_counts), device=local_counts.device, dtype=local_counts.dtype)
            expected_expert_id = (indices * probs).sum().item()
        
        loss["Most Frequently Activated Expert"] = winner_id
        loss["Expected Activated Expert"] = expected_expert_id

        # """ Uniform Dist Regularization for Gating """
        # gating_sum = gating.sum(dim=0)                 # (E,)
        # local_B = torch.tensor([B], device=self.device, dtype=torch.float32)

        # p_bar = gating_sum / local_B.clamp_min(1.0)          # (E,)
        # p_bar = p_bar.clamp_min(1e-8)

        # # KL(p_bar || uniform)
        # log_u = -math.log(E)
        # kl = (p_bar * (p_bar.log() - log_u)).sum()

        self.velocity_loss_ema = self.velocity_loss_ema_factor * self.velocity_loss_ema + (1.0 - self.velocity_loss_ema_factor) * (moe_loss.detach().clone().item() + 0.2 * sinkhorn_loss.detach().clone().item()) \
                                 if not None else moe_loss.detach().clone().item() + 0.2 * sinkhorn_loss.detach().clone().item()
        
        #loss["Total"] = moe_loss + 0.2 * sinkhorn_loss +  max(math.pow(0.1, iterations/13115), 0.1) * self.velocity_loss_ema * kl
        #loss["EMA_prior_posterior"] = torch.pow(prior_cls_token - posterior_target.detach(), 2).mean()
        #loss["Gating_Uniform"] = kl.detach().clone().item()
        

        loss["Total"] = moe_loss + 0.2 * sinkhorn_loss + kl
        loss["prior_posterior"] = kl.detach().clone().item()
        loss["velocity"] = moe_loss.detach().clone().item()
        loss["Sinkhorn"] = sinkhorn_loss.detach().clone().item()

        return loss 




    def train_step(self, data: dict[str, Any], epoch, total_epochs, iterations) -> dict[str, Any]:
        self._ready_train()
        self._zero_grad()

        loss = self.forward(data, epoch, total_epochs, iterations)

        self._backward(loss)
        self._step()
        
        # try:
        #     if self.posterior_ema is not None:
        #         # IMPORTANT: update EMA after optimizer step such that it's when the weights have been synced across GPUs
        #         with torch.no_grad():
        #             self.update_posterior_ema()
        # except:
        #     pass
        detached_loss = self._detached_loss(loss)
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
