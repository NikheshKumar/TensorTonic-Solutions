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
    _, _, dv = value.shape

    outputs = []
    
    for i in range(S):
        q = query[:, i:i+1, :]
        k = key[:, :i+1, :]
        v = value[:, :i+1, :]

        scores = q @ k.transpose(-2,-1) / (dk ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        att = weights @ v

        outputs.append(att)

    outputs = torch.cat(outputs, dim=1)
    
    return outputs, k, v
