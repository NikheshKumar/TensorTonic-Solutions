import torch

def pool_to_budget(patch_features: torch.Tensor, budget: int) -> tuple[torch.Tensor, int]:
    """
    Returns a tuple of the pooled soft tokens and the window size.
    """
    G_h, G_w, D = patch_features.shape

    for k in range(1, max(G_w, G_h)+1):
        if (-(-G_h//k))*(-(-G_w//k)) <= budget:
            break

    pooled = torch.zeros((-(-G_h//k) * -(-G_w//k), D), dtype=patch_features.dtype, device=patch_features.device)

    n_h = -(-G_h//k)
    n_w = -(-G_w//k)
    
    for a in range(n_h):
        for b in range(n_w):
            patch = patch_features[a*k:a*k+k, b*k:b*k + k]
            pooled[a * n_w + b] = patch.mean(dim=(0, 1))


    return (pooled, k)

    
        

