import torch
import torch.nn as nn
import torch.nn.functional as F


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