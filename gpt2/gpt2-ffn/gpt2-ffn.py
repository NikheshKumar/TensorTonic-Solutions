import torch

def ffn(x: torch.Tensor, W1: torch.Tensor, b1: torch.Tensor, W2: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    """
    Returns: torch.Tensor of same shape as x after FFN with GELU activation
    """
    def gelu(x):
        e = torch.erf(x/(2.0**0.5))
        return x * 0.5 * (1+e)

    return gelu(x @ W1.T + b1)@W2.T + b2