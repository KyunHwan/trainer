import copy
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from schedulefree.radam_schedulefree import RAdamScheduleFree as radam_free
from schedulefree.adamw_schedulefree import AdamWScheduleFree as adamw_free

class Action_Critic_Trainer(nn.Module):
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

        self.q_target = self._make_target(self.models['q_function']).to(self.device)
        self.q_function_processor_target = self._make_target(self.models['q_function_processor']).to(self.device)

        # self.flow_matching_policy = flow_matching_policy
        # self.noise_latent_actor = noise_latent_actor

        self.loss_func = nn.MSELoss()

    def forward(self, data, stats):

        # Data preparation
        future_input_data = {
            'head': None,
            'left': None,
            'right': None,
            'proprio': None,
            'action': None,
        }
        cur_input_data = {
            'head': None,
            'left': None,
            'right': None,
            'proprio': None,
            'action': None,
        }
        with torch.no_grad():
            head_feat, _ = self.models['backbone'](data['observation.images.cam_head'][:, 0, :])
            left_feat, _ = self.models['backbone'](data['observation.images.cam_left'][:, 0, :])
            right_feat, _ = self.models['backbone'](data['observation.images.cam_right'][:, 0, :])
            future_input_data['head'] = head_feat
            future_input_data['left'] = left_feat
            future_input_data['right'] = right_feat
            future_input_data['proprio'] = (data['observation.state'][:, :50, :] - stats['observation.state']['mean']) / (stats['observation.state']['std'] + 1e-8)

            _head_feat, _ = self.models['backbone'](data['observation.images.cam_head'][:, 1, :])
            _left_feat, _ = self.models['backbone'](data['observation.images.cam_left'][:, 1, :])
            _right_feat, _ = self.models['backbone'](data['observation.images.cam_right'][:, 1, :])
            cur_input_data['head'] = _head_feat
            cur_input_data['left'] = _left_feat
            cur_input_data['right'] = _right_feat
            cur_input_data['proprio'] = (data['observation.state'][:, 50:, :] - stats['observation.state']['mean']) / (stats['observation.state']['std'] + 1e-8)
            cur_input_data['action'] = (data['action'] - stats['action']['mean']) / (stats['action']['std'] + 1e-8)

        Q_target_output = None
        with torch.no_grad():
            processor_output = self.models['noise_processor'](future_input_data)
            noise = self.models['noise_actor'](processor_output)
            action_chunk = self.models['openpi_model'](observation = {
                                                                'proprio': data['observation.state'].detach()[:, 0, :].detach().cpu().to(torch.float32).numpy(),
                                                                'head': data['observation.images.cam_head'][:, 0, :].detach().cpu().to(torch.float32).numpy(),
                                                                'left': data['observation.images.cam_left'][:, 0, :].detach().cpu().to(torch.float32).numpy(),
                                                                'right': data['observation.images.cam_right'][:, 0, :].detach().cpu().to(torch.float32).numpy(),
                                                                'prompt': data['prompt']
                                                              }, 
                                                        noise=noise.detach().cpu().to(torch.float32).numpy())
            future_input_data['action'] = (action_chunk.to(torch.float32) - stats['action']['mean']) / (stats['action']['std'] + 1e-8)

            Q_target_output = self.q_target(self.q_function_processor_target(future_input_data))

        # Q-chunking
        reward_tensor = data['labels.reward']
        action_chunk_size = reward_tensor.shape[1]
        # 1. Create the exponents: [1, 2, ..., action_chunk_size]
        # Ensure it lives on the same device (CPU/GPU) as your reward tensor
        powers = torch.arange(1, action_chunk_size + 1, device=reward_tensor.device).to(torch.float32)

        # 2. Calculate the discount factors: [gamma^1, gamma^2, ..., gamma^h]
        discounts = self.discount_factor ** powers

        # 3. Multiply and aggregate along the chunk dimension (dim=1)
        # The resulting shape will be (batch,)
        data['reward_chunk'] = torch.sum(reward_tensor * discounts, dim=1).reshape(reward_tensor.shape[0], 1)

        loss = self.loss_func(self.models['q_function'](self.models['q_function_processor'](cur_input_data)),\
                              data['reward_chunk'] + (self.discount_factor ** 50) * Q_target_output)
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
        q_source_net = self._unwrap_model(self.models['q_function'])
        for p_src, p_tgt in zip(q_source_net.parameters(), self.q_target.parameters()):
            p_tgt.data.mul_(1.0 - self.target_update_rate).add_(self.target_update_rate * p_src.data)

        q_preprocessor_source_net = self._unwrap_model(self.models['q_function_processor'])
        for p_src, p_tgt in zip(q_preprocessor_source_net.parameters(), self.q_function_processor_target.parameters()):
            p_tgt.data.mul_(1.0 - self.target_update_rate).add_(self.target_update_rate * p_src.data)