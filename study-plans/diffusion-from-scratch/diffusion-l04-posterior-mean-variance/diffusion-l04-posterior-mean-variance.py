import torch

def posterior_mean_variance(x0: torch.Tensor, x_t: torch.Tensor, timesteps: torch.Tensor, betas: torch.Tensor) -> dict:
    """
    Returns a dict with the posterior "mean" (same shape as x0) and the per-sample "variance" and "log_variance" of shape (B,).
    """
    alpha = 1.0 - betas
    alpha_bars = torch.cumprod(alpha, dim=0)

    alpha_bars_adj = torch.cat([torch.ones_like(alpha_bars[:1]), alpha_bars[:-1]])
    
    beta_tilde_unindexed = ((1.0 - alpha_bars_adj) * betas / (1.0 - alpha_bars))
    beta_tilde = beta_tilde_unindexed[timesteps]
    
    log_var = torch.log(beta_tilde)
    
    term1 = ((betas * torch.sqrt(alpha_bars_adj)) / (1.0 - alpha_bars))[timesteps].view(-1, *[1] * (x0.dim() - 1)) * x0
    term2 = ((1.0 - alpha_bars_adj)*torch.sqrt(alpha)/(1.0 - alpha_bars))[timesteps].view(-1, *[1] * (x0.dim() - 1)) * x_t
    mu = term1 + term2

    log_var = torch.where(timesteps>=1, torch.log(beta_tilde), torch.log(beta_tilde_unindexed[1]))
    
    return {"mean":mu, "variance":beta_tilde, "log_variance":log_var}

    
    
