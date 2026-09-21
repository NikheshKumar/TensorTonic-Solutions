import math
import torch

def glm_attention(
    x: torch.Tensor,
    W_qkv: torch.Tensor,
    b_qkv: torch.Tensor,
    W_o: torch.Tensor,
    qk_gamma_q: torch.Tensor,
    qk_gamma_k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_attention_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rope_dim: int,
    eps_rms: float = 1e-5,
) -> torch.Tensor:
    """
    Returns the float64 causal attention output with shape (n_tokens, d_model).
    """

    y_squeeze_bool = False
    if x.dim()==2:
        y_squeeze_bool = True
        x = x.unsqueeze(0)
        
    qkv = x @ W_qkv + b_qkv
    Q, K, V = torch.split(qkv, [num_attention_heads * head_dim, num_kv_heads * head_dim, num_kv_heads * head_dim], dim=-1)

    B, seq_q, _ = Q.shape
    seq_k = K.shape[1]


    Q = Q.reshape(B, seq_q, num_attention_heads, head_dim).transpose(1, 2)
    K = K.reshape(B, seq_k, num_kv_heads, head_dim).transpose(1, 2)
    V = V.reshape(B, seq_k, num_kv_heads, head_dim).transpose(1, 2)


    Q_norm = qk_gamma_q * Q * torch.rsqrt((Q**2).mean(-1, keepdim=True) + eps_rms)
    K_norm = qk_gamma_k * K * torch.rsqrt((K**2).mean(-1, keepdim=True) + eps_rms)


    def apply_half_rope(z, sin, cos):
        out = z.clone()
        z_rot = z[..., :rope_dim]
        z_first  = z_rot[..., :rope_dim // 2]
        z_second = z_rot[...,rope_dim // 2:]
        out[..., :rope_dim // 2] = z_first * cos - z_second * sin
        out[..., rope_dim // 2:rope_dim] = z_first * sin + z_second * cos
        return out

    Q_norm = apply_half_rope(Q_norm, sin, cos)
    K_norm = apply_half_rope(K_norm, sin, cos)

    
    K_norm = K_norm.repeat_interleave(num_attention_heads // num_kv_heads, dim=1)   
    V = V.repeat_interleave(num_attention_heads // num_kv_heads, dim=1)        

    scores = (Q_norm @ K_norm.transpose(-2, -1)) / (head_dim ** 0.5)

    mask = torch.triu(torch.ones(seq_q, seq_k, dtype=torch.bool, device=scores.device),diagonal=1)
    
    scores = scores.masked_fill(mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    
    att = weights @ V                                

    context = att.transpose(1, 2).contiguous().reshape(B, seq_q, num_attention_heads * head_dim)
    
    y = context @ W_o   

    if y_squeeze_bool:
        y = y.squeeze(0)

    return y