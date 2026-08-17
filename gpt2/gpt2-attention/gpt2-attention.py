import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V):
    """
    Returns: torch.Tensor of shape (batch, seq_q, d_v)
    """

    Q = torch.tensor(Q, dtype=torch.float32)
    K = torch.tensor(K, dtype=torch.float32)
    V = torch.tensor(V, dtype=torch.float32)
    
    B, seq_q, d_q = Q.shape
    _, seq_k, d_k = K.shape
    _, seq_v, d_v = V.shape


    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)

    weights = F.softmax(scores, dim=-1)

    att = weights @ V

    return att
