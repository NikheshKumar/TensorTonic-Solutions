import torch

def noise_prediction_loss(x0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor, alpha_bars: torch.Tensor, weight: torch.Tensor, time_weight: torch.Tensor, bias: torch.Tensor) -> dict:
    """
    Returns a dict with the batch loss as a Python float under "loss" and the per-sample losses of shape (B,) under "per_sample_loss".
    """

    B = noise.shape[0]

    x0 = x0.reshape(B, -1)
    noise = noise.reshape(B, -1)

    x = torch.sqrt(alpha_bars[timesteps].reshape(B,1)) * x0 + torch.sqrt(1.0 - alpha_bars[timesteps].reshape(B,1)) * noise

    noise_hat = x @ weight + timesteps.reshape(B,1).float() * time_weight / len(alpha_bars) + bias

    per_sample_loss = torch.mean((noise_hat - noise) **2, dim=-1)

    L = torch.mean(per_sample_loss).item()

    return {"loss":L, "per_sample_loss":per_sample_loss}