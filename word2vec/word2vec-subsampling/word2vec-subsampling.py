import torch

def subsample_keep_probs(counts: torch.Tensor, t: float = 1e-5) -> torch.Tensor:
    """
    Returns torch.Tensor of shape (vocab_size,) with the keep-probability for each word.
    """
    # YOUR CODE HERE
    total = torch.sum(counts)
    f = counts / total
    p = torch.clamp(torch.sqrt(t/f), max=1.0)

    return p
