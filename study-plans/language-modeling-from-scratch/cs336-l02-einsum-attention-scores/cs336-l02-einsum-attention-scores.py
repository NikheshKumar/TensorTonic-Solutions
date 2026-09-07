import torch

def attention_scores(q, k, num_heads):
    """
    Returns: tensor of shape (batch, heads, query_length, key_length)
    """
    B, S_q, D = q.shape
    _, S_k, _ = k.shape

    d_h = D // num_heads

    q = q.reshape(B, S_q, num_heads, d_h).transpose(1,2)
    k = k.reshape(B, S_k, num_heads, d_h).transpose(1,2)

    scores = q @ k.transpose(-2,-1) / (d_h**0.5)

    return scores
