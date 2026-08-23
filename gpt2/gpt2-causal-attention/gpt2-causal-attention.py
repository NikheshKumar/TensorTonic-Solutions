import torch
import torch.nn.functional as F
import math

def causal_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns: torch.Tensor of shape (batch, seq_len, d_v)
    """
    # Your code here
    B, seq_len, d_q = Q.shape

    _, _, d_k = K.shape

    scores = Q @ K.transpose(-2,-1) / math.sqrt(d_k)

    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

    scores = scores.masked_fill(mask, value=-float("inf"))

    weights = torch.softmax(scores, dim=-1)

    att = weights @ V

    return att