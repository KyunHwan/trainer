import torch 
import torch.nn as nn
from schedulefree.radam_schedulefree import RAdamScheduleFree as radam_free
from schedulefree.adamw_schedulefree import AdamWScheduleFree as adamw_free

class Noise_Latent_Actor_Trainer(nn.Module):
    def __init__(self, 
                 models,
                 device,
                #  noise_latent_critic, 
                #  noise_latent_actor
                ):
        super().__init__()
        self.models = models
        self.device = device
        # self.noise_latent_Q = noise_latent_critic
        # self.noise_latent_actor = noise_latent_actor


    def forward(self, data, stats):
        # need to prevent optimization to occur to noise_latent_critic such that backpropagation
        # doesn't happen for noise_latent_critic. 
        # and only allow backpropagation to happen for noise_latent_actor, 
        # such that only the weights for noise_latent_actor are updated

        # Data preparation
        cur_input_data = {
            'head': None,
            'left': None,
            'right': None,
            'proprio': None,
            'action': None,
        }
        feats = self.models['noise_actor_img_encoder']({
            'head': data['observation.images.cam_head'][:, 1, :],
            'left': data['observation.images.cam_left'][:, 1, :],
            'right': data['observation.images.cam_right'][:, 1, :],
        })
        cur_input_data['head'] = feats['head']
        cur_input_data['left'] = feats['left']
        cur_input_data['right'] = feats['right']
        with torch.no_grad():
            cur_input_data['proprio'] = (data['observation.state'][:, 50:, :] - stats['observation.state']['mean']) / (stats['observation.state']['std'] + 1e-8)

        cur_input_data['action'] = self.models['noise_actor'](self.models['noise_processor'](cur_input_data))
        loss = self.models['noise_q_function'](self.models['noise_q_function_processor'](cur_input_data)) * -1.0

        return loss.mean()