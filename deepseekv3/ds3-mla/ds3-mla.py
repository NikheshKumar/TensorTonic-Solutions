import torch
import math

def mla(x: torch.Tensor, W_dkv: torch.Tensor, W_uk: torch.Tensor, W_uv: torch.Tensor,
        W_q: torch.Tensor, W_qr: torch.Tensor, W_kr: torch.Tensor, W_o: torch.Tensor,
        cos_freq: torch.Tensor, sin_freq: torch.Tensor, num_heads: int, eps: float = 1e-6) -> torch.Tensor:
    """
    Returns: torch.Tensor of shape (batch, seq_len, d_model)
    """
    # YOUR CODE HERE
    B, seq_len, d_model = x.shape
    d_nope = W_uk.shape[0] // num_heads
    d_rope = W_qr.shape[0] // num_heads
    d_v = W_uv.shape[0] // num_heads
    

    c_kv = x @ W_dkv.T

    K_nope = (c_kv @ W_uk.T).reshape(B, seq_len, num_heads, d_nope).transpose(1,2)

    V = (c_kv @ W_uv.T).reshape(B, seq_len, num_heads, d_v).transpose(1,2)

    Q_nope = (x @ W_q.T).reshape(B, seq_len, num_heads, d_nope).transpose(1,2)

    def apply_rope(z, sin_freq, cos_freq):
        
        sin_freq = sin_freq[:seq_len].unsqueeze(0).unsqueeze(0)
        cos_freq = cos_freq[:seq_len].unsqueeze(0).unsqueeze(0)

        z1 = z[..., :z.shape[-1]//2]
        z2 = z[..., z.shape[-1]//2: ]

        z_rotated = torch.cat([z1*cos_freq - z2 * sin_freq, z1 * sin_freq + z2 * cos_freq], dim=-1)

        return z_rotated

    Q_rope = (x @ W_qr.T).reshape(B, seq_len, num_heads, d_rope).transpose(1, 2)

    K_rope = (x @ W_kr.T).reshape(B, seq_len, num_heads, d_rope).transpose(1, 2)

    Q_rope = apply_rope(Q_rope, sin_freq, cos_freq)

    K_rope = apply_rope(K_rope, sin_freq, cos_freq)

    Q = torch.cat([Q_nope, Q_rope], dim=-1)

    K = torch.cat([K_nope, K_rope], dim=-1)

    d_h = Q.shape[-1]

    scores = Q @ K.transpose(-2, -1) / (d_h**0.5)


    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=Q.device), diagonal=1)
    scores = scores.masked_fill(mask, value=-float("inf"))
    
    
    weights = torch.softmax(scores, dim=-1)

    att = weights @ V

    context = att.transpose(1,2).contiguous().reshape(B, seq_len, d_v * num_heads)

    output = x + context @ W_o.T

    return output