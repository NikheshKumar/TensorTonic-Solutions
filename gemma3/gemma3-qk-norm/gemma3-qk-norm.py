import torch

def qk_norm(Q: torch.Tensor, K: torch.Tensor, gamma_q: torch.Tensor, gamma_k: torch.Tensor, eps: float = 1e-6):
    """
    Returns: tuple (Q_norm, K_norm) each of shape (batch, heads, seq_len, d_head)
    """
    # YOUR CODE HERE

    Q_norm = Q * gamma_q * torch.rsqrt(torch.mean(Q**2, dim=-1, keepdim=True) + eps)
    K_norm = K * gamma_k * torch.rsqrt(torch.mean(K**2, dim=-1, keepdim=True) + eps)

    return (Q_norm, K_norm)