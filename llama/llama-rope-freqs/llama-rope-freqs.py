import torch

def precompute_rope_freqs(max_seq_len, d_head, base=10000.0):
    """
    Returns: tuple of (cos_table, sin_table) both shape (max_seq_len, d_head//2)
    """
    # YOUR CODE HERE
    i = torch.arange(0, d_head//2)

    theta = base**(-2*i/d_head)

    positions = torch.arange(0, max_seq_len, dtype=torch.float32)

    angle = positions.unsqueeze(1) * theta.unsqueeze(0)

    cos_table = torch.cos(angle)
    sin_table = torch.sin(angle)


    return cos_table, sin_table