import torch

def gated_attention(x, W_q, W_k, W_v, W_o, W_gate, gamma_norm, num_heads, eps=1e-6):
    """
    Returns: torch.Tensor of shape (batch, seq, d_model)
    """
    # YOUR CODE HERE

    B, seq_len, d_model = x.shape
    d_k = d_model // num_heads
    
    x_rms = x * gamma_norm * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)

    Q = (x_rms @ W_q.T).view(B, seq_len, num_heads, d_k).transpose(1, 2)
    K = (x_rms @ W_k.T).view(B, seq_len, num_heads, d_k).transpose(1, 2)
    V = (x_rms @ W_v.T).view(B, seq_len, num_heads, d_k).transpose(1, 2)

    scores = Q @ K.transpose(-2,-1) / (d_k**0.5)
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
    scores = scores.masked_fill(mask, value=-float('inf'))

    weights = torch.softmax(scores, dim=-1)

    att = (weights @ V).transpose(1, 2).contiguous().view(B, seq_len, d_model)

    gate = torch.sigmoid(x_rms @ W_gate.T)

    output = x + gate * (att @ W_o.T)

    return output
    