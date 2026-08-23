import torch
import torch.nn.functional as F
import math

def multi_head_attention(x: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor, W_o: torch.Tensor, n_heads: int) -> torch.Tensor:
    """
    Returns: torch.Tensor of shape (batch, seq_len, d_model)
    """
    # Your code here

    B, seq_len, d_model = x.shape
    d_head = d_model // n_heads

    Q = (x @ W_q).reshape(B, seq_len, n_heads, d_head).transpose(1,2)
    K = (x @ W_k).reshape(B, seq_len, n_heads, d_head).transpose(1,2)
    V = (x @ W_v).reshape(B, seq_len, n_heads, d_head).transpose(1,2)

    scores = Q @ K.transpose(-2,-1) / math.sqrt(d_head)

    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

    scores = scores.masked_fill(mask, value=-float("inf"))

    weights = torch.nn.functional.softmax(scores, dim=-1)

    att = weights @ V

    concat = att.transpose(1,2).contiguous().reshape(B, seq_len, d_model)

    output = concat @ W_o

    return output