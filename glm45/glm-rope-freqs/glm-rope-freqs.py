import torch

def rope_freqs(n_tokens: int, rope_dim: int, rope_theta: float) -> dict[str, torch.Tensor]:
    """
    Returns a dictionary containing float64 cos and sin tables.
    """
    i = torch.arange(0, rope_dim//2)

    theta = rope_theta**(-2.0*i/rope_dim)

    pos = torch.arange(0,n_tokens)

    angles = pos.unsqueeze(-1) * theta

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return {"cos": cos, "sin": sin}

    

    
    