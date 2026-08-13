import torch
from typing import Tuple

def apply_rotary_position_embeddings(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    base: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns: (rotated query tensor, rotated key tensor), same shapes as inputs
    """

    d_k = query.shape[-1]

    i = torch.arange(0, d_k, 2, device=query.device) 
    theta = base**(-i/d_k)        
    freqs = positions[..., None] * theta

    def apply_rope(x, cos_freq, sin_freq):
    
        x_rotated = x.clone()
    
        x_even = x[...,0::2]
        x_odd = x[...,1::2]

        while cos_freq.ndim < x.ndim:
            cos_freq = cos_freq.unsqueeze(0)
            sin_freq = sin_freq.unsqueeze(0)
    
        x_rotated[...,0::2] = x_even * cos_freq - x_odd * sin_freq
        x_rotated[...,1::2] = x_even * sin_freq + x_odd * cos_freq
    
        return x_rotated

    cos_freq = torch.cos(freqs)
    sin_freq = torch.sin(freqs)

    q_rotated = apply_rope(query, cos_freq, sin_freq)

    k_rotated = apply_rope(key, cos_freq, sin_freq)

    return (q_rotated, k_rotated)