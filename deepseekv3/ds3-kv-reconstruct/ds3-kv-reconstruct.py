import torch

def kv_reconstruct(c_kv: torch.Tensor, W_uk: torch.Tensor, W_uv: torch.Tensor, num_heads: int):
    """
    Returns: tuple (K, V) with K shape (batch, heads, seq, d_nope) and V shape (batch, heads, seq, d_head)
    """
    # YOUR CODE HERE
    B, S, _ = c_kv.shape
    
    d_nope = W_uk.shape[0] // num_heads

    d_h = W_uv.shape[0] // num_heads
    
    K = (c_kv @ W_uk.T).reshape(B, S, num_heads, d_nope).transpose(1,2)

    V = (c_kv @ W_uv.T).reshape(B, S, num_heads, d_h).transpose(1,2)

    return (K, V)