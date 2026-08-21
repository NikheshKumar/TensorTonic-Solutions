import torch
import math

def sandwich_norm(x: torch.Tensor, sub_out: torch.Tensor, gamma_post: torch.Tensor,
                  layer_idx: int, eps: float = 1e-6) -> torch.Tensor:
    """
    Apply post-norm with depth scaling and residual connection.
    Returns: torch.Tensor of shape (batch, seq, d_model)
    """
    # YOUR CODE HERE
    
    def apply_rms_norm(z, gamma, eps):
        return z * gamma * torch.rsqrt(torch.mean(z**2, dim=-1, keepdim=True) + eps)
        
    post_normed = apply_rms_norm(sub_out, gamma_post, eps)

    depth_scale = 1.0/math.sqrt(2.0 * layer_idx + 1)

    output = x +  post_normed * depth_scale

    return output