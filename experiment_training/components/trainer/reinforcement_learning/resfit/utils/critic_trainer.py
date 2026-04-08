import copy
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from schedulefree.radam_schedulefree import RAdamScheduleFree as radam_free
from schedulefree.adamw_schedulefree import AdamWScheduleFree as adamw_free

class Critic_Trainer(nn.Module):
    def __init__(self, 
                 models,
                 device,
                 target_update_rate: float=0.005,
                 discount_factor: float=0.99,
                 ):
        # takes in noise critic & action critic 
        super().__init__()
        self.models = models
        self.device = device
        self.target_update_rate = target_update_rate
        self.discount_factor = discount_factor

        self.q_target = self._make_target(self.models['resfit_q_function']).to(self.device)

        self.loss_func = nn.MSELoss()

    def forward(self, data, stats):
        # resfit_q_function
        # resfit_residual_actor

        with torch.no_grad():
            # get future base_policy_action & future state
            future_base_policy_action = data['base_policy_action'][:, 3, :]  # (B, 24)
            future_state = {
                'head': data['observation.images.cam_head'][:, 0, :],
                'left': data['observation.images.cam_left'][:, 0, :,],
                'right': data['observation.images.cam_right'][:, 0, :],
                'proprio': data['observation.state'][:, 0, :].unsqueeze(1)  # (B, 1, 24)
            }

            # calculate residual action for future state
            future_residual_action = self.models['resfit_residual_actor'](
                state=future_state, action=future_base_policy_action.unsqueeze(1))  # (B, 1, 24)

            # get updated future action = future residual action + future base_policy_action
            future_action = future_residual_action + data['base_policy_action'][:, 3, :]

            future_data = {
                'head': future_state['head'],
                'left': future_state['left'],
                'right': future_state['right'],
                'proprio': future_state['proprio'],
                'action': future_action.unsqueeze(1)  # (B, 1, 24)
            }

            future_val = self.q_target(data=future_data, subsample_q=True)

            # calculate TD n=3 target reward value
            # TD
            reward_tensor = data['labels.reward']
            reward_horizon = reward_tensor.shape[1]
            # 1. Create the exponents: [1, 2, ..., reward_horizon]
            # Ensure it lives on the same device (CPU/GPU) as your reward tensor
            powers = torch.arange(0, reward_horizon, device=reward_tensor.device).to(torch.float32)

            # 2. Calculate the discount factors: [gamma^1, gamma^2, ..., gamma^h]
            discounts = self.discount_factor ** powers

            # 3. Multiply and aggregate along the chunk dimension (dim=1)
            # The resulting shape will be (batch,)
            data['reward_chunk'] = torch.sum(reward_tensor * discounts, dim=1).reshape(reward_tensor.shape[0], 1)

            

        loss = self.loss_func(self.models['resfit_q_function'](
                                data={
                                    'head': data['observation.images.cam_head'][:, 1, :],
                                    'left': data['observation.images.cam_left'][:, 1, :,],
                                    'right': data['observation.images.cam_right'][:, 1, :],
                                    'proprio': data['observation.state'][:, 1, :].unsqueeze(1),  # (B, 1, 24)
                                    'action': data['action'][:, 0, :].unsqueeze(1)               # (B, 1, 24)
                                },
                                subsample_q=False,
                                critic=True
                              ),
                              data['reward_chunk'] + (self.discount_factor ** reward_horizon) * future_val)
        return loss
        


    def _unwrap_model(self, net):
        """Helper function to strip DDP or Ray wrappers."""
        # DDP and most Ray wrappers expose the base model via the .module attribute
        if hasattr(net, "module"):
            return net.module
        return net

    def _make_target(self, net):
        # 1. Unwrap before copying to avoid duplicating distributed hooks/locks
        base_net = self._unwrap_model(net)
        
        tgt = copy.deepcopy(base_net)
        tgt.eval()                              # no dropout/bn updates in eval mode
        for p in tgt.parameters():
            p.requires_grad_(False)             # do not track grads
        
        # Note: You generally want to move the target network to the right device here 
        # if it isn't already, e.g., tgt.to(next(base_net.parameters()).device)
        return tgt

    @torch.no_grad()
    def update_target(self) -> None:
        """Polyak averaging: target <- tau*source + (1-tau)*target."""
        q_source_net = self._unwrap_model(self.models['resfit_q_function'])
        for p_src, p_tgt in zip(q_source_net.parameters(), self.q_target.parameters()):
            p_tgt.data.mul_(1.0 - self.target_update_rate).add_(self.target_update_rate * p_src.data)