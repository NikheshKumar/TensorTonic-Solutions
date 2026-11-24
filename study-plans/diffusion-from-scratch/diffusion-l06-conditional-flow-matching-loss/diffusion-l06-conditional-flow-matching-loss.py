import torch

def flow_matching_loss(x_data: torch.Tensor, x_noise: torch.Tensor, t: torch.Tensor, weight: torch.Tensor, time_weight: torch.Tensor, bias: torch.Tensor) -> dict:
    """
    Returns a dict with "x_t", "target", "per_sample_loss" (tensors) and "loss" (Python float).
    """

    x_t = (1.0-t[:, None]) * x_data + t[:, None] * x_noise

    u = x_noise - x_data

    v = x_t @ weight + t[:, None]*time_weight + bias

    per_sample_loss = torch.mean((v-u)**2, dim=1)

    l = float(torch.mean(per_sample_loss, dim=0))

    return {"x_t":x_t, "target":u, "per_sample_loss":per_sample_loss, "loss":l}
    
