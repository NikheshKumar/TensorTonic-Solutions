import torch

def layernorm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Returns: torch.Tensor with LayerNorm applied across the last dimension
    """
    m = torch.mean(x, dim=-1, keepdim=True)

    var = torch.var(x, dim=-1, keepdim=True, unbiased=False)

    return gamma * (x - m) * torch.rsqrt(var + eps) + beta