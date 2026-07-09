import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    
    batch_size, seq_len_q, d_k = Q.shape
    
    scores = torch.matmul(Q, K.transpose(-2, -1))
    scaled_scores = scores / math.sqrt(d_k)
    weights = F.softmax(scaled_scores, dim=-1)

    scaled_att = weights @ V

    return scaled_att
    