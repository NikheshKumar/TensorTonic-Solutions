import torch

def causal_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Returns the attention tensor with the same shape and dtype as q.
    """

    B, d_q, S_q, D = q.shape
    _, d_kv, S_k, _ = k.shape

    num_heads = d_q // d_kv

    q = q.view(B, d_kv, num_heads, S_q, D)
    k = k.view(B, d_kv, 1, S_k, D)
    v = v.view(B, d_kv, 1, S_k, D)

    scores = q @ k.transpose(-2,-1) / (D**0.5)

    mask = torch.triu(torch.ones(S_q, S_k, dtype=torch.bool, device=q.device), diagonal=(S_k- S_q) + 1)

    scores = scores.masked_fill(mask, value=-float("inf"))

    weights = torch.softmax(scores, dim=-1)

    att = weights @ v  

    return att.reshape(B, d_q, S_q, D).to(q.dtype)
