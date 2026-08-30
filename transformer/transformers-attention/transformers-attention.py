import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    B, seq_len, d_model = Q.shape

    scores = Q @ K.transpose(-2,-1) / math.sqrt(d_model)

    weights = F.softmax(scores, dim=-1)

    att = weights @ V

    return att