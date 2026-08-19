import torch

def softmax_with_sinks(scores: torch.Tensor, sinks: torch.Tensor) -> torch.Tensor:
    """Returns: torch.Tensor of shape (H, n_q, n_k) with attention weights that sum to less than 1."""
    # YOUR CODE HERE
    H, n_q, n_k = scores.shape

    sinks = sinks[:, None, None].expand(-1, n_q, -1)

    scores = torch.cat([scores, sinks], dim=-1)

    max_val = torch.max(scores, dim=-1, keepdim=True).values

    num = torch.exp(scores-max_val)

    den = torch.sum(num, dim=-1, keepdim=True)

    return num[...,:-1] / den
