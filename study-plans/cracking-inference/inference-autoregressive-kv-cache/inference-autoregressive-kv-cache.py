import torch

def cached_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple:
    """
    Returns (outputs, key_cache, value_cache), float32 tensors in sequence order.
    """

    B, S, dk = query.shape

    scores = query @ key.transpose(-2, -1) / (dk ** 0.5)   
    mask = torch.triu(torch.ones((S, S), dtype=torch.bool, device=query.device), diagonal=1)
    scores = scores.masked_fill(mask, float('-inf'))

    weights = torch.softmax(scores, dim=-1)    
    
    outputs = weights @ value                              

    return outputs, key, value