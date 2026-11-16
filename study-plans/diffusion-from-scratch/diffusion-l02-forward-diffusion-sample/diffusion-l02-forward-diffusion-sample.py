import torch

def forward_diffusion_sample(x0: torch.Tensor, timesteps: torch.Tensor, alpha_bars: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """
    Returns the noisy samples x_t with the same shape and dtype as x0.
    """
    alpha_bars = alpha_bars.view(-1, *([1] * (x0.dim() - 1)))
    x = torch.sqrt(alpha_bars[timesteps]) * x0 + torch.sqrt(1.0 - alpha_bars[timesteps]) * noise

    return x
