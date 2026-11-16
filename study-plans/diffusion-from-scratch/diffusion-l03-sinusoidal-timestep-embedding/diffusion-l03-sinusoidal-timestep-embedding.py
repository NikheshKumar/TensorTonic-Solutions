import torch
import math

def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: float) -> torch.Tensor:
    """
    Returns the float32 sinusoidal embedding of shape (B, dim).
    """

    idx = torch.arange(0, dim//2)
    f = torch.exp(-math.log(max_period)*idx / (dim//2))

    args = timesteps[:, None].float() * f[None, :]

    e = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    if dim%2:
        e = torch.nn.functional.pad(e, (0, 1))

    return e
