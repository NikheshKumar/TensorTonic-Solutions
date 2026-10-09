import torch

def topk_accuracy(logits: list, targets: list, k: int) -> float:
    """
    Returns the top-k accuracy as a float, rounded to 4 decimals.
    """
    logits = torch.tensor(logits, dtype=torch.float64)
    targets = torch.tensor(targets, dtype=torch.long)

    if targets.ndim == 2:
        targets = targets.argmax(dim=1)

    k = min(k, logits.shape[1])
    
    idx = torch.argsort(logits, dim=1, descending=True, stable=True)
    idx = idx[:, :k]
    c = (idx == targets.unsqueeze(1)).any(dim=1)

    return round(c.float().mean().item(), 4)

    