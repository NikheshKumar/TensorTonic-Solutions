import torch

def decoupled_rope(k_nope: torch.Tensor, k_rope_input: torch.Tensor,
                   cos_freq: torch.Tensor, sin_freq: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    Returns: torch.Tensor of shape (batch, heads, seq, d_nope + d_rope)
    """
    # YOUR CODE HERE

    B, H, S, d_nope = k_nope.shape
    d_rope = k_rope_input.shape[-1] // num_heads

    def apply_rope(z, sin_freq, cos_freq):

        sin_freq = sin_freq[:S].unsqueeze(0).unsqueeze(0)
        cos_freq = cos_freq[:S].unsqueeze(0).unsqueeze(0)

        z1 = z[..., : z.shape[-1]//2]
        z2 = z[..., z.shape[-1]//2 : ]

        z_rotated = torch.cat([z1 * cos_freq - z2 * sin_freq, z1 * sin_freq +  z2 * cos_freq], dim=-1)
        
        return z_rotated
        
    
    k_rope = k_rope_input.view(B, S, num_heads, d_rope)

    k_rope = k_rope.permute(0, 2, 1, 3)
    
    k_rope = apply_rope(k_rope, sin_freq, cos_freq)

    K_full = torch.cat([k_nope, k_rope], dim=-1)

    return K_full