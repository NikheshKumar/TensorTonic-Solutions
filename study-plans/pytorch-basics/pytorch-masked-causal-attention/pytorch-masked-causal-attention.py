import torch

def causal_attention(Q, K, V):
    """
    Returns: masked attention output tensor
    """
    import math 

    B, S_q, d_k = Q.shape
    _, S_k, _ = K.shape

    M = torch.triu(torch.full((S_q, S_k), -float('inf'), device=Q.device), diagonal=1)

    scores = (Q @ K.transpose(-2,-1) / math.sqrt(d_k)) + M

    weights = torch.softmax(scores, dim=-1)

    att = weights @ V


    return att