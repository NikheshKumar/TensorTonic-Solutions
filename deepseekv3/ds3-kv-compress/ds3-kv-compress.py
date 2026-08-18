import torch

def kv_compress(x: torch.Tensor, W_dkv: torch.Tensor) -> torch.Tensor:
    """
    Returns: torch.Tensor of shape (batch, seq_len, d_c)
    """
    # YOUR CODE HERE
    c_kv = x @ W_dkv.T

    return c_kv 