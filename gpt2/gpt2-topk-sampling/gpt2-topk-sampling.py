import torch

def apply_temperature(logits, temperature):
    """
    Returns: torch.Tensor of scaled logits
    """
    # YOUR CODE HERE
    scaled_logits = logits / temperature
    return scaled_logits

def top_k_filter(logits, k):
    """
    Returns: torch.Tensor with non-top-k values set to -inf
    """
    # YOUR CODE HERE
    filtered = logits.clone()
    values, indices = torch.topk(logits, k) 
    threshold = values[-1]
    mask = logits < threshold
    filtered[mask] =-float("inf")
    return filtered

def sample_from_logits(logits, random_val):
    """
    Returns: int (sampled token id)
    """
    # YOUR CODE HERE
    p = torch.softmax(logits, dim=-1)
    cdf = torch.cumsum(p, dim=-1)
    token_id = int(torch.searchsorted(cdf, torch.tensor(random_val, dtype=cdf.dtype)).item())
    return token_id