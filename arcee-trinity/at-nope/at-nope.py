import torch
import math

def nope_attention(x: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor,
                   W_v: torch.Tensor, W_o: torch.Tensor, gamma_norm: torch.Tensor,
                   num_heads: int, eps: float = 1e-6) -> torch.Tensor:
    """
    Standard causal self-attention with RMSNorm but NO positional encoding.
    Returns: torch.Tensor of shape (batch, seq, d_model)
    """
    # YOUR CODE HERE
    B, seq_len, d_model = x.shape
    d_k = d_model // num_heads
    
    x_rms = x * gamma_norm * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)

    Q = (x_rms @ W_q.T).reshape(B, seq_len, num_heads, d_k).transpose(1,2) 
    K = (x_rms @ W_k.T).reshape(B, seq_len, num_heads, d_k).transpose(1,2)
    V = (x_rms @ W_v.T).reshape(B, seq_len, num_heads, d_k).transpose(1,2)

    scores = Q @ K.transpose(-2,-1) / (d_k**0.5)
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
    scores = scores.masked_fill(mask, value=-float("inf"))
    weights = torch.softmax(scores, dim=-1)

    att = (weights @ V).transpose(1,2).contiguous().reshape(B, seq_len, d_model)

    output = x + att @ W_o.T

    return output