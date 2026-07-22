import torch
from typing import Optional

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Returns: attention output tensor of shape (batch, seq_q, d_v)
    """

    B, S_q, d_k = query.shape
    _, S_k, _ = key.shape
    
    scores = query @ key.transpose(-2,-1) / (d_k**0.5)

    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    
    weights = torch.softmax(scores, dim=-1)

    att = weights @ value

    return att
