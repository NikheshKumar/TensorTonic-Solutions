import torch

def rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Returns: Normalized tensor of same shape as x
    """
    # YOUR CODE HERE
    num = x * gamma
    den = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)

    return num / den