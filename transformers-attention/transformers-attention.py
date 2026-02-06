import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here

    _,_,d = Q.shape
    scores = torch.matmul(Q,K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d, dtype=Q.dtype, device=Q.device))
    weights = torch.softmax(scores, dim=-1)
    result = torch.matmul(weights, V)

    return result


