import torch

def sliding_window_mask(n: int, window: int) -> torch.Tensor:
    """Returns: torch.Tensor of shape (n, n), float, with 0 in visible positions and -inf in masked positions."""
    # YOUR CODE HERE
    mask = torch.zeros((n,n), dtype=torch.float64)

    row = torch.arange(n)[:, None] 
    col = torch.arange(n)[None,:]

    visible_region = (row >= col)

    if window==0:
        mask = mask.masked_fill(~visible_region, value=-float("inf"))
        return mask

    visible_region = visible_region & (row - col < window)
    mask = mask.masked_fill(~visible_region, value=-float("inf"))

    return mask
