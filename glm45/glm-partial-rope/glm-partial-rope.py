import torch

def partial_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_dim: int,
) -> dict[str, torch.Tensor]:
    """
    Returns a dictionary containing q_rotated and k_rotated tensors.
    """
    def apply_half_rope(z, sin, cos):
        
        out = z.clone()
        z_rot = z[...,:rope_dim]
        
        z_first = z_rot[...,:rope_dim//2]
        z_second = z_rot[...,rope_dim//2:]

        if cos.dim() < z_first.dim():
            cos = cos.unsqueeze(-2)
            sin = sin.unsqueeze(-2)
        
        out[..., :rope_dim//2] = z_first * cos - z_second * sin
        out[..., rope_dim//2:rope_dim] = z_first * sin + z_second * cos

        return out

    q_rot = apply_half_rope(q, sin, cos)
    k_rot = apply_half_rope(k, sin, cos)

    return {"q_rotated": q_rot, "k_rotated": k_rot}
        