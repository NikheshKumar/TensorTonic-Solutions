import torch

def apply_rope(q, k, freqs_cos, freqs_sin):
    """
    Returns: tuple of (q_rotated, k_rotated) same shapes as input
    """
    # YOUR CODE HERE
    q_rotated = q.clone()
    k_rotated = k.clone()

    q_even = q[...,0::2]
    q_odd = q[...,1::2]

    k_even = k[...,0::2]
    k_odd = k[...,1::2]
    
    q_rotated[..., 0::2] = q_even * freqs_cos - q_odd * freqs_sin
    q_rotated[..., 1::2] = q_even * freqs_sin + q_odd * freqs_cos

    k_rotated[..., 0::2] = k_even * freqs_cos - k_odd * freqs_sin
    k_rotated[..., 1::2] = k_even * freqs_sin + k_odd * freqs_cos

    return (q_rotated, k_rotated)