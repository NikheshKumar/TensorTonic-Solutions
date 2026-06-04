import torch
import torch.nn.functional as F
import math

def grouped_query_attention(x: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor,
                            W_v: torch.Tensor, W_o: torch.Tensor,
                            n_heads: int, n_kv_heads: int) -> torch.Tensor:
    """
    Returns: (batch, seq_len, d_model)
    """
    # YOUR CODE HERE
    
    batch, seq_len, d_model = x.shape
    d_head = d_model // n_heads
    
    Q = x @ W_q.T
    K = x @ W_k.T
    V = x @ W_v.T

    Q = Q.view(batch, seq_len, n_heads, d_head).transpose(1, 2)
    K = K.view(batch, seq_len, n_kv_heads, d_head).transpose(1, 2)
    V = V.view(batch, seq_len, n_kv_heads, d_head).transpose(1, 2)

    num_queries = n_heads // n_kv_heads

    K = K.unsqueeze(2).repeat(1, 1, num_queries, 1, 1).view(batch, n_heads, seq_len, d_head)
    V = V.unsqueeze(2).repeat(1, 1, num_queries, 1, 1).view(batch, n_heads, seq_len, d_head)

    scores = Q @ K.transpose(-2, -1) / (d_head**0.5)

    attn = F.softmax(scores, dim=-1)

    context = (attn @ V).transpose(1, 2).contiguous()
    context = context.view(batch, seq_len, d_model)

    output = context @ W_o.T

    return output