import math
import torch
import torch.nn as nn
from geomloss import SamplesLoss
from trainer.trainer.registry import LOSS_BUILDER_REGISTRY

@LOSS_BUILDER_REGISTRY.register("sinkhorn_knopp")
class SinkhornKnoppFactory:
    def __init__(self, 
                 p: int,
                 lam_state: float,
                 blur: float,
                 debias: bool,          
                 backend: str,
                 scaling: float,):
        self.p = p
        self.lam_state = lam_state
        self.blur = blur
        self.debias = debias
        self.backend = backend
        self.scaling = scaling
    def build(self) -> nn.Module:
        return KOTSinkhornLoss(
            p=self.p,
            lam_state=self.lam_state,
            blur=self.blur,
            debias=self.debias,          # closer to “Sinkhorn OT” in VFP wording
            backend=self.backend,
            scaling=self.scaling,
            )
    
class KOTSinkhornLoss(nn.Module):
    def __init__(self, p: int, lam_state: float, blur: float, debias: bool,
                 backend: str, scaling: float):
        super().__init__()
        self.lam_state = lam_state

        # Build an actual squared-distance cost (NOT 1/2 * squared)
        if backend == "tensorized":
            def sqdist_cost(x, y):
                # x: (B,N,D), y: (B,M,D) -> (B,N,M)
                return ((x[:, :, None, :] - y[:, None, :, :]) ** 2).sum(dim=-1)
            cost = sqdist_cost
        else:
            cost = "(SqDist(X,Y) / IntCst(2))"  # KeOps formula (no /2)

        self.ot = SamplesLoss(
            loss="sinkhorn",
            p=2,              # keep p=2 for Sinkhorn settings; cost override controls exact form
            blur=blur,
            debias=debias,
            cost=None,        # <-- IMPORTANT
            backend=backend,
            scaling=scaling,
        )

    def forward(self, pred_action, target_action, state_pred, state_target):
        B = pred_action.shape[0]
        pred_a = pred_action.reshape(B, -1)
        targ_a = target_action.reshape(B, -1)

        pred_s = state_pred.reshape(B, -1)
        targ_s = state_target.reshape(B, -1)

        scale = math.sqrt(self.lam_state)
        x = torch.cat([pred_a, scale * pred_s], dim=-1)
        y = torch.cat([targ_a, scale * targ_s], dim=-1)

        return self.ot(x, y)

# class KOTSinkhornLoss(nn.Module):
#     """
#     VFP-style K-OT regularizer:
#       OT between two point clouds of (state, action_chunk) pairs
#       with ground cost ||a-a'||^2 + lam_state*||s-s'||^2

#     Expected shapes (common case):
#       pred_action   : (B, H, A) or (B, Da)
#       target_action : (B, H, A) or (B, Da)
#       state_pred    : (B, Ds)
#       state_target  : (B, Ds)

#     Computes ONE OT distance per minibatch (scalar).
#     """
#     def __init__(
#         self,
#         p: int,
#         lam_state: float,
#         blur: float,
#         debias: bool,          # closer to “Sinkhorn OT” in VFP wording
#         backend: str,
#         scaling: float,
#     ):
#         super().__init__()
#         self.lam_state = lam_state
#         # We want squared Euclidean cost, not the default p=2 cost (which is 0.5*||·||^2).
#         # For tensorized backend we can supply a python cost; for KeOps backends we'd use "SqDist(X,Y)".
#         if backend == "tensorized":
#             def sqdist_cost(x, y):
#                 # x: (B,N,D), y: (B,M,D) -> (B,N,M)
#                 return ((x[:, :, None, :] - y[:, None, :, :]) ** 2).sum(dim=-1)
#             cost = sqdist_cost
#         else:
#             cost = "(SqDist(X,Y) / IntCst(2))"  # KeOps formula (no /2)

#         self.ot = SamplesLoss(
#             loss="sinkhorn",
#             p=2,
#             blur=blur,
#             debias=debias,
#             cost=None,
#             backend=backend,
#             scaling=scaling,
#         )

#     def forward(self, pred_action, target_action, state_pred, state_target):
#         B = pred_action.shape[0]
#         # Flatten action chunks so each chunk is ONE point in R^(H*A)
#         pred_a = pred_action.reshape(B, -1)
#         targ_a = target_action.reshape(B, -1)

#         # Flatten states to (B, Ds)
#         pred_s = state_pred.reshape(B, -1)
#         targ_s = state_target.reshape(B, -1)

#         # Augment features so squared distance equals ||a-a'||^2 + lam*||s-s'||^2
#         scale = math.sqrt(self.lam_state)
#         x = torch.cat([pred_a, scale * pred_s], dim=-1)   # (B, D_aug)
#         y = torch.cat([targ_a, scale * targ_s], dim=-1)   # (B, D_aug)

#         # One OT problem between the two sets of B points
#         return self.ot(x, y)  # scalar
