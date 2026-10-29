import torch

def glm_mtp(
    h_prev: torch.Tensor,
    next_token_embed: torch.Tensor,
    norm_h_g: torch.Tensor,
    norm_e_g: torch.Tensor,
    proj_W: torch.Tensor,
    final_norm_g: torch.Tensor,
    lm_head_W: torch.Tensor,
    eps_rms: float = 1e-5,
) -> torch.Tensor:
    """
    Returns float64 MTP logits with shape (n_tokens, vocab_size).
    """
    def rms_norm(z, gamma, eps):
        y = (z**2).mean(dim=-1, keepdim=True)
        return gamma * z * torch.rsqrt(y + eps)

    h_prev = torch.as_tensor(h_prev, dtype=torch.float64)
    next_token_embed = torch.as_tensor(next_token_embed, dtype=torch.float64)
    norm_h_g = torch.as_tensor(norm_h_g, dtype=torch.float64)
    norm_e_g = torch.as_tensor(norm_e_g, dtype=torch.float64)
    proj_W = torch.as_tensor(proj_W, dtype=torch.float64)
    final_norm_g = torch.as_tensor(final_norm_g, dtype=torch.float64)
    lm_head_W = torch.as_tensor(lm_head_W, dtype=torch.float64)
    
    h_hat = rms_norm(h_prev, norm_h_g, eps_rms)
    
    e_hat = rms_norm(next_token_embed, norm_e_g, eps_rms)

    z = torch.cat([h_hat, e_hat], dim=-1) @ proj_W

    z_hat = rms_norm(z, final_norm_g, eps_rms)

    logits = z_hat @ lm_head_W

    return logits.to(torch.float64)

    
        