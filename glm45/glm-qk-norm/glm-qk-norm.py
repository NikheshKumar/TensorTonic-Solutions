import torch

def qk_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    gamma_q: torch.Tensor,
    gamma_k: torch.Tensor,
    eps: float = 1e-5,
) -> dict[str, torch.Tensor]:
    """
    Returns a dictionary containing q_norm and k_norm tensors.
    """
    def f(z, gamma, eps):
        return gamma * z * torch.rsqrt(z.square().mean(dim=-1, keepdim=True) + eps)

    q_norm = f(q, gamma_q, eps)
    k_norm = f(k, gamma_k, eps)

    return {"q_norm":q_norm, "k_norm":k_norm}