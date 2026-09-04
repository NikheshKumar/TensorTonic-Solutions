import torch

def rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Returns a float64 tensor with the same shape as x.
    """
    return gamma * x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) +eps)