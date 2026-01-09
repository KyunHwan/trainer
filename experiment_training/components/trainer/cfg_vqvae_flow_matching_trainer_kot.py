import torch
import torch.nn as nn
import einops

from flow_matching.path.affine import AffineProbPath
from flow_matching.path.scheduler.scheduler import CondOTScheduler

import torch.nn.functional as F

from offline_trainer.registry import TRAINER_REGISTRY
from typing import Any
import copy


@TRAINER_REGISTRY.register("cfg_vqvae_flow_matching_trainer_kot")
class CFG_VQVAE_Flow_Matching_Trainer(nn.Module):
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
        self.posterior_ema = copy.deepcopy(self.unwrap(self.models["vqvae_posterior"])).to(device).eval()
        for p in self.posterior_ema.parameters():
            p.requires_grad_(False)
        self.posterior_ema_factor = 0.999

    def forward(self, data: dict[str, Any], epoch, total_epochs) -> dict[str, torch.Tensor]:
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

        """ VQVAE Codebook """
        codebook_output = self.models['vqvae_codebook'](continuous_vec=posterior_cls_token, train=True, replacement=False)
        related_codebook_quantized_vec = codebook_output['q']
        loss['codebook_min_dist'] = codebook_output['codebook_min_dist']
        loss['codebook_max_dist'] = codebook_output['codebook_max_dist']

        """ NSVQ """
        # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9696322
        noise_vec = torch.randn_like(posterior_cls_token, device=posterior_cls_token.device)
        normalized_noise_vec = F.normalize(noise_vec, p=2, dim=-1)
        distance_magnitude = torch.norm(posterior_cls_token - related_codebook_quantized_vec, p=2, dim=-1, keepdim=True)

        # Classifier-Free Guidance
        bernoulli = torch.bernoulli(torch.tensor([1 - ((0.5 * min(epoch, 2)) / 2) for i in range(posterior_cls_token.shape[0])]))
        bernoulli = bernoulli.view(*([bernoulli.shape[0]] + [1]*(posterior_cls_token.ndim - 1))).to(posterior_cls_token.device, dtype=posterior_cls_token.dtype)

        simulated_quantized_vec = (posterior_cls_token + normalized_noise_vec * distance_magnitude) * bernoulli
         
        # """ Info Encoder """
        # conditioning_info = self.models['info_encoder'](cond_proprio=data['observation.state'],
        #                                                 cond_visual=torch.cat([head_image_features, 
        #                                                                        left_image_features, 
        #                                                                        right_image_features],
        #                                                                        axis=1),
        #                                                 cond_semantic=simulated_quantized_vec)
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
                                                                         conditioning_info], dim=1),
                                                 discrete_semantic_input=simulated_quantized_vec,)
        
        # EMA
        # For prior training --> since posterior is a moving target, naively doing l2 distance between the two destabilizes prior learning 
        with torch.no_grad():
            self.posterior_ema.eval()
            posterior_target = self.posterior_ema(cond_proprio=data['observation.state'],
                                                  cond_visual=head_image_features,
                                                  cond_semantic=head_image_semantic,
                                                  action = data['action']
                                                 )  # e.g., returns posterior_cls_token_ema
            loss["True_prior_posterior"] = torch.sum(torch.pow(prior_cls_token.detach().clone() - posterior_cls_token.detach().clone(), 2), dim=-1, keepdim=False).mean().item()
        
        err = (dx_t - dx_t_hat).pow(2)
        velocity_loss = err.view(err.shape[0], -1).sum(dim=1).mean()
    
        sinkhorn_loss = self.loss(pred_action = noise + self.models['action_decoder'](
                                                            time = torch.zeros_like(time), 
                                                            noise = noise, 
                                                            memory_input = conditioning_info.detach(),
                                                            discrete_semantic_input=simulated_quantized_vec,), 
                                     target_action = data['action'], 
                                     state_pred = data['observation.state'], 
                                     state_target = data['observation.state'])
        
        # Enforces the latent vectors to commit to respective vectors in the codebook
        commitment_loss = (posterior_cls_token - related_codebook_quantized_vec.detach()).pow(2).view(-1, posterior_cls_token.shape[-1]).sum(dim=-1).mean()
        
        loss["Total"] = velocity_loss + 0.2 * sinkhorn_loss + 0.25 * commitment_loss
        loss["EMA_prior_posterior"] = torch.sum(torch.pow(prior_cls_token - posterior_target, 2).view(-1, prior_cls_token.shape[-1]), dim=-1, keepdim=False).mean()
        loss["velocity"] = velocity_loss.detach().clone().item()
        loss["Sinkhorn"] = sinkhorn_loss.detach().clone().item()
        loss["posterior_codebook"] = commitment_loss.detach().clone().item()
            
        return loss, posterior_cls_token.detach().clone()




    def train_step(self, data: dict[str, Any], epoch, total_epochs, iterations) -> dict[str, Any]:
        self._ready_train()
        self._zero_grad()

        loss, continuous_vec = self.forward(data, epoch, total_epochs)

        self._backward(loss)
        self._step()
        
        try:
            if self.posterior_ema is not None:
                # IMPORTANT: update EMA after optimizer step such that it's when the weights have been synced across GPUs
                with torch.no_grad():
                    self.update_posterior_ema()
        except:
            pass
        detached_loss = self._detached_loss(loss)
        if epoch < 2 and iterations % ((epoch + 1) * 10) == 0:
            with torch.no_grad():
                output = self.models['vqvae_codebook'](continuous_vec=continuous_vec, train=True, replacement=True)
                self._reset_opt_state_rows(output['dead_indices'])
            detached_loss['num_vecs_replaced'] = output['num_vecs_replaced']
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
    
    @torch.no_grad()
    def _reset_opt_state_rows(self, row_indices: torch.Tensor | None):
        """
        Zero Adam/AdamW moments for specific rows of a 2D parameter (e.g., nn.Embedding.weight).

        optimizer: torch.optim.Adam or AdamW (works for most Adam variants using exp_avg/exp_avg_sq)
        param: the Parameter object whose state you want to reset (e.g., model.vq_codebook.weight)
        row_indices: 1D LongTensor of row indices to reset (dead_indices)
        """
        if row_indices is None or row_indices.numel() == 0:
            return
        if self.models['vqvae_codebook'].parameters() not in self.optimizers['vqvae_codebook'].state:
            return  # state not initialized yet (e.g., before first optimizer.step())

        st = self.optimizers['vqvae_codebook'].state[self.models['vqvae_codebook'].parameters()]
        # Indices must be on the same device as the state tensors for index_fill_
        def _zero_rows_(t: torch.Tensor):
            if not torch.is_tensor(t):
                return
            if t.ndim < 2:
                return  # e.g., scalar step tensor; ignore
            idx = row_indices.to(device=t.device, dtype=torch.long)
            # zero the rows along dim 0
            t.index_fill_(0, idx, 0)

        # Standard Adam/AdamW keys
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq", "z"):
            if key in st:
                _zero_rows_(st[key])
        
