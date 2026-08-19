import torch
import math

def compute_yarn_rope_freqs(head_dim: int, base: float, initial_context_length: int, scaling_factor: float, ntk_alpha: float = 1.0, ntk_beta: float = 32.0):
    """Returns: (concentration: float, inv_freq: torch.Tensor of shape (head_dim/2,))"""
    # YOUR CODE HERE

    i = torch.arange(0,head_dim,2, dtype=torch.float64)
    
    freqs = base**(i/head_dim)

    inv_freqs = 1.0 / freqs

    low = head_dim * math.log(initial_context_length / (ntk_beta * 2.0 * torch.pi)) /(2.0 * math.log(base))

    high = head_dim * math.log(initial_context_length / (ntk_alpha * 2.0 * torch.pi)) /(2.0 * math.log(base))

    j = torch.arange(0,head_dim//2, 1, dtype=torch.float64)

    ramp = (j-low) / (high-low)

    mask = 1.0 - torch.clip(ramp,0.0,1.0)

    final_freqs = (1-mask) * inv_freqs / scaling_factor + mask * inv_freqs

    concentration = 0.1 * math.log(scaling_factor) + 1.0

    if scaling_factor <= 1.0 :

        concentration = 1.0      

        final_freqs = inv_freqs


    return (concentration, final_freqs)

    

    
