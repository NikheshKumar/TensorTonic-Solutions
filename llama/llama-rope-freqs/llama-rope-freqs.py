import torch

def precompute_rope_freqs(max_seq_len, d_head, base=10000.0):
    """
    Returns: tuple of (cos_table, sin_table) both shape (max_seq_len, d_head//2)
    """
    # YOUR CODE HERE
    i = torch.arange(0, d_head, 2, dtype=torch.float32)
    theta = base**(-i/d_head)
    p = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(p, theta)


    cos_table = torch.cos(angles)
    sin_table = torch.sin(angles)


    return cos_table, sin_table