import torch

def scaled_dot_product_attention(Q, K, V):
    """
    Returns: attention output tensor
    """
    _, _, d_k = Q.shape
    
    scores = Q @ K.transpose(-2,-1) / float(d_k**0.5)

    weights = torch.softmax(scores, dim=-1)

    attn = weights @ V

    return attn