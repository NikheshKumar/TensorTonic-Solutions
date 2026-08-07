import torch

def repeat_kv(kv: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Returns: (batch, n_kv_heads * n_rep, seq_len, d_head)
    """
    # YOUR CODE HERE
    if n_rep==1:
        return kv
        
    batch, n_kv_heads, seq_len, d_head = kv.shape
    kv_head_rep = kv.repeat_interleave(n_rep, dim=1)

    return kv_head_rep.reshape(batch, n_kv_heads * n_rep, seq_len, d_head)
    