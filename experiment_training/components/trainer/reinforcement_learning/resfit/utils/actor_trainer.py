import torch 
import torch.nn as nn
from schedulefree.radam_schedulefree import RAdamScheduleFree as radam_free
from schedulefree.adamw_schedulefree import AdamWScheduleFree as adamw_free

class Actor_Trainer(nn.Module):
    def __init__(self, 
                 models,
                 device,
                ):
        super().__init__()
        self.models = models
        self.device = device


    def forward(self, data, stats):
        # resfit_residual_actor
        # resfit_q_function

        # calculate maximum of mean of q function with respect to residual model

        # Data preparation

        base_policy_action = data['base_policy_action'][:, 0, :]  # (B, 24)

        cur_input_state = {
            'head': data['observation.images.cam_head'][:, 1, :],
            'left': data['observation.images.cam_left'][:, 1, :],
            'right': data['observation.images.cam_right'][:, 1, :],
            'proprio': data['observation.state'][:, 1, :].unsqueeze(1),  # (B, 1, 24)
        }

        cur_input_data = {}
        cur_input_data['action'] = self.models['resfit_residual_actor'](
            state=cur_input_state,
            action=base_policy_action.unsqueeze(1)  # (B, 1, 24) for preprocessor
        ) + base_policy_action  # (B, 24) + (B, 24)

        cur_input_data['action'] = cur_input_data['action'].unsqueeze(1)  # (B, 1, 24) for Q preprocessor
        cur_input_data.update(cur_input_state)

        self.models['resfit_q_function'].eval()
        loss = -1.0 * self.models['resfit_q_function'](
            data=cur_input_data,
            subsample_q=False,
            critic=False
        )

        return loss.mean()