import torch
from typing import Tuple

def route_tokens_to_experts(
    router_logits: torch.Tensor,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns: (expert_indices, routing_weights), each of shape (num_tokens, top_k)
    """
    values, indices = torch.sort(router_logits, dim=-1, descending=True, stable=True)

    values = values[..., :top_k]
    indices = indices[..., :top_k]

    weights = torch.softmax(values, dim=-1)

    return indices, weights
