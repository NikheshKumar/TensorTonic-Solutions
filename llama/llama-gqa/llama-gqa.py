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
    B, seq_len, d_model = x.shape
    d_head = d_model // n_heads
    
    Q = (x @ W_q.T).reshape(B, seq_len, n_heads, d_head).transpose(1, 2)
    K = (x @ W_k.T).reshape(B, seq_len, n_kv_heads, d_head).transpose(1, 2)
    V = (x @ W_v.T).reshape(B, seq_len, n_kv_heads, d_head).transpose(1, 2)

    num_queries = n_heads // n_kv_heads

    K = K.repeat_interleave(num_queries, dim=1)
    V = V.repeat_interleave(num_queries, dim=1)

    scores = Q @ K.transpose(-2,-1) / math.sqrt(d_head)
    att = F.softmax(scores, dim=-1)
    context = (att @ V).transpose(1, 2).contiguous().reshape(B, seq_len, d_model)

    output = context @ W_o.T

    return output