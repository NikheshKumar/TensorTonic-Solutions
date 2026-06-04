import torch

def apply_rope(q, k, freqs_cos, freqs_sin):
    """
    Returns: tuple of (q_rotated, k_rotated) same shapes as input
    """
    # YOUR CODE HERE

    freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(0)
    freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(0)

    q_rotated = torch.zeros_like(q)
    q_rotated[..., 0::2] = q[..., 0::2] * freqs_cos - q[..., 1::2] * freqs_sin
    q_rotated[..., 1::2] = q[..., 0::2] * freqs_sin + q[..., 1::2] * freqs_cos

    k_rotated = torch.zeros_like(k)
    k_rotated[..., 0::2] = k[..., 0::2] * freqs_cos - k[..., 1::2] * freqs_sin
    k_rotated[..., 1::2] = k[..., 0::2] * freqs_sin + k[..., 1::2] * freqs_cos
    

    return (q_rotated, k_rotated)