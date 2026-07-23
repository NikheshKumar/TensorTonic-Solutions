import torch
from typing import Tuple

def multi_head_latent_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_down: torch.Tensor,
    w_up_k: torch.Tensor,
    w_up_v: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns: (output tensor of shape (batch, seq, d_model), latent tensor of shape (batch, seq, d_latent))
    """
    B, S_q, d_model = hidden_states.shape
    d_k = d_model // num_heads

    c = hidden_states @ w_down
    K = c @ w_up_k
    V = c @ w_up_v
    Q = hidden_states @ w_q

    q = Q.reshape(B, S_q, num_heads, d_k).transpose(1,2)
    k = K.reshape(B, S_q, num_heads, d_k).transpose(1,2)
    v = V.reshape(B, S_q, num_heads, d_k).transpose(1,2)

    scores = q @ k.transpose(-2,-1) / (d_k**0.5)
    

    if causal:
        mask = torch.triu(torch.ones(S_q, S_q, dtype=torch.bool, device=hidden_states.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))


    weights = torch.softmax(scores, dim=-1)

    att = weights @ v


    mla = att.transpose(1,2).contiguous().reshape(B, S_q, d_model) @ w_o

    return mla, c
