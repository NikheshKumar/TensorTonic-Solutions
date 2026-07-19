import torch

def causal_attention(Q, K, V):
    """
    Returns: masked attention output tensor
    """
    import math 

    B, S_q, d_k = Q.shape
    _, S_k, _ = K.shape

    scores = (Q @ K.transpose(-2,-1) / math.sqrt(d_k))

    M = torch.triu(torch.ones(S_q, S_k, dtype=torch.bool, device=Q.device), diagonal=1)

    scores.masked_fill_(M, -float('inf'))

    weights = torch.softmax(scores, dim=-1)

    att = weights @ V


    return att