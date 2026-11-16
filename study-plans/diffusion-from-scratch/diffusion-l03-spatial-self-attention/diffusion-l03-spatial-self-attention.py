import math
import torch
import torch.nn.functional as F

def spatial_self_attention(x: torch.Tensor, params: dict, num_groups: int, num_heads: int) -> torch.Tensor:
    """
    Returns x plus the attention output, shape (B, C, H, W) with the dtype of x.
    """
    out_dtype = x.dtype
    B, C, H, W = x.shape
    n = num_heads
    d = C // n

    norm_weight, norm_bias = params.get("norm_weight"), params.get("norm_bias")
    q_weight, k_weight, v_weight, out_weight = params.get("q_weight"), params.get("k_weight"), params.get("v_weight"), params.get("out_weight")
    q_bias, k_bias, v_bias, out_bias = params.get("q_bias"), params.get("k_bias"), params.get("v_bias"), params.get("out_bias")

    def GN(x, W, b):
        return F.group_norm(x, num_groups, W, b, eps=1e-5)

    Z = GN(x, norm_weight, norm_bias).flatten(2).transpose(1, 2)

    def split_transpose(z):
        return z.reshape(B, H*W, n, d).transpose(1,2)

    Q = split_transpose(Z @ q_weight.T + q_bias)
    K = split_transpose(Z @ k_weight.T + k_bias)
    V = split_transpose(Z @ v_weight.T + v_bias)

    scores = Q @ K.transpose(-2,-1) / (d**0.5)
    weights = F.softmax(scores, dim=-1)

    att = (weights @ V).transpose(1,2).reshape(B, H*W, C)

    out = x + (att.contiguous() @ out_weight.T + out_bias).transpose(1,2).reshape(B, C, H, W)

    return out.to(dtype=out_dtype)
