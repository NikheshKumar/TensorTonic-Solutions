import torch

def gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Returns: torch.Tensor with GELU applied element-wise
    """
    e = torch.erf(x/(2.0**0.5))

    return x * 0.5 * (1 + e)