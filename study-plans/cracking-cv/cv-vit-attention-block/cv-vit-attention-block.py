import torch

def vit_attention(x: list, W_qkv: list, b_qkv: list, W_o: list, b_o: list, num_heads: int) -> list:
    """
    Returns a float list of shape (B, N, D), rounded to 4 decimals.
    """
    x = torch.tensor(x, dtype=torch.float64)
    W_qkv = torch.tensor(W_qkv, dtype=torch.float64)
    b_qkv = torch.tensor(b_qkv, dtype=torch.float64)
    W_o = torch.tensor(W_o, dtype=torch.float64)
    b_o = torch.tensor(b_o, dtype=torch.float64)


    B, N, D = x.shape
    d_head = D // num_heads

    def split_heads(z):
        z = z.view(B, N, num_heads, d_head).transpose(1,2)
        return z
        
    qkv = x @ W_qkv + b_qkv
    q, k, v = torch.chunk(qkv, 3, dim=-1)

    q = split_heads(q)
    k = split_heads(k)
    v = split_heads(v)

    scores = q @ k.transpose(-2,-1) / (d_head**0.5)

    weights = torch.softmax(scores, dim=-1)

    att = weights @ v

    y = att.transpose(1,2).reshape(B,N,D) @ W_o + b_o

    return torch.round(y, decimals=4).tolist()

        