import torch
import torch.nn as nn
from schedulefree.radam_schedulefree import RAdamScheduleFree as radam_free
from schedulefree.adamw_schedulefree import AdamWScheduleFree as adamw_free

class Noise_Latent_Critic_Trainer(nn.Module):
    def __init__(self, 
                 models,
                 device,
                #  action_critic, 
                #  noise_latent_critic,
                #  flow_matching_policy
                ):
        # takes in noise critic & action critic 
        super().__init__()
        self.models = models
        self.device = device
        # self.Q = action_critic
        # self.noise_latent_Q = noise_latent_critic
        # self.flow_matching_policy = flow_matching_policy

        self.loss_func = nn.MSELoss()

    def forward(self, data, stats):


        # Data preparation
        input_data = {
            'head': None,
            'left': None,
            'right': None,
            'proprio': None,
            'action': None,
        }
        with torch.no_grad():
            head_feat, _ = self.models['backbone'](data['observation.images.cam_head'][:, 1, :])
            left_feat, _ = self.models['backbone'](data['observation.images.cam_left'][:, 1, :])
            right_feat, _ = self.models['backbone'](data['observation.images.cam_right'][:, 1, :])
            input_data['head'] = head_feat
            input_data['left'] = left_feat
            input_data['right'] = right_feat
            input_data['proprio'] = (data['observation.state'][:, 50:, :] - stats['observation.state']['mean']) / (stats['observation.state']['std'] + 1e-8)

        Q_output = None
        batch, _, _ = data['action'].shape
        noise = torch.randn(batch, 50, 32)
        with torch.no_grad():
            action_chunk = self.models['openpi_model'](observation = {
                                                                'proprio': data['observation.state'].detach()[:, 50, :].detach().cpu().to(torch.float32).numpy(),
                                                                'head': data['observation.images.cam_head'][:, 1, :].detach().cpu().to(torch.float32).numpy(),
                                                                'left': data['observation.images.cam_left'][:, 1, :].detach().cpu().to(torch.float32).numpy(),
                                                                'right': data['observation.images.cam_right'][:, 1, :].detach().cpu().to(torch.float32).numpy(),
                                                                'prompt': data['prompt']
                                                              }, 
                                                        noise=noise.to(torch.float32).numpy())
            input_data['action'] = (action_chunk.to(torch.float32) - stats['action']['mean']) / (stats['action']['std'] + 1e-8)

            Q_output = self.models['q_function'](self.models['q_function_processor'](input_data))
        
        input_data['action'] = noise.to(self.device)
        loss = self.loss_func(self.models['noise_q_function'](self.models['noise_q_function_processor'](input_data)), 
                              Q_output)
        return loss