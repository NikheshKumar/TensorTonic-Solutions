import torch

def moe_router(x: torch.Tensor, gate_weight: torch.Tensor, top_k: int):
    """
    Returns: tuple (top_k_indices, top_k_weights)
      - top_k_indices: (batch, seq_len, top_k) integer indices of selected experts
      - top_k_weights: (batch, seq_len, top_k) renormalized softmax weights
    """
    # YOUR CODE HERE
    p = torch.softmax(x@gate_weight.T, dim=-1)

    top_k_weights, top_k_indices = torch.topk(p, top_k, dim=-1)

    top_k_weights = top_k_weights / torch.sum(top_k_weights, dim=-1, keepdim=True)

    return (top_k_indices, top_k_weights)