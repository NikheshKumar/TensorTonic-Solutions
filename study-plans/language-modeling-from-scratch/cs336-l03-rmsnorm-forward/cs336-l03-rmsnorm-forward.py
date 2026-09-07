import torch

def rmsnorm(x, g, epsilon):
    """
    Returns: RMS-normalized tensor
    """
    x_copy = torch.tensor(x, dtype=torch.float32)
    g = torch.tensor(g, dtype=torch.float32)

    return (x_copy * g * torch.rsqrt(torch.mean(x_copy**2, dim=-1, keepdim=True) + epsilon)).to(x.dtype)

    
