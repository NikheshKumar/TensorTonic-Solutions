import torch
import math

def gpt2_decoder_block(x, gamma1, beta1, W_q, W_k, W_v, W_o, gamma2, beta2, W1, b1, W2, b2, n_heads):
    """
    Returns: nested list of shape (seq_len, d_model), rounded to 4 decimals."""
    
    def layernorm(x, gamma, beta):
        eps = 1e-5
        m = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
        x_norm = (x-m) * torch.rsqrt(var + eps)

        return gamma * x_norm + beta

    def gelu(z):
        e = torch.erf(z / (2.0**0.5))
        return z * 0.5 * (1.0+e)

    x = torch.tensor(x, dtype=torch.float64)
    
    gamma1 = torch.tensor(gamma1, dtype=torch.float64)
    beta1 = torch.tensor(beta1, dtype=torch.float64)
    gamma2 = torch.tensor(gamma2, dtype=torch.float64)
    beta2 = torch.tensor(beta2, dtype=torch.float64)
    
    W_q = torch.tensor(W_q, dtype=torch.float64)
    W_k = torch.tensor(W_k, dtype=torch.float64)
    W_v = torch.tensor(W_v, dtype=torch.float64)
    W_o = torch.tensor(W_o, dtype=torch.float64)
    
    W1 = torch.tensor(W1, dtype=torch.float64)
    b1 = torch.tensor(b1, dtype=torch.float64)
    W2 = torch.tensor(W2, dtype=torch.float64)
    b2 = torch.tensor(b2, dtype=torch.float64)
    
    x_new = layernorm(x, gamma1, beta1)

    seq_len, d_model = x.shape
    d_k = d_model // n_heads

    Q = (x_new @ W_q).reshape(seq_len, n_heads, d_k).transpose(0, 1)
    K = (x_new @ W_k).reshape(seq_len, n_heads, d_k).transpose(0, 1)
    V = (x_new @ W_v).reshape(seq_len, n_heads, d_k).transpose(0, 1)
    
    scores = Q @ K.transpose(-2,-1) / (d_k ** 0.5)

    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

    scores = scores.masked_fill(mask, value=-float('inf'))

    weights = torch.softmax(scores, dim=-1)

    att = (weights @ V).permute(1,0,2).contiguous().reshape(seq_len, d_model)

    x_prime = x + att @ W_o

    ffn = gelu(layernorm(x_prime, gamma2, beta2) @ W1 + b1) @ W2 + b2

    out = x_prime + ffn

    return [[round(v.item(), 4) for v in row] for row in out]
    