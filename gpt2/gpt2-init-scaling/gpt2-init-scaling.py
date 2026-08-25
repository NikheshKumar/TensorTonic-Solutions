import torch
import math

def scale_residual_weights(W, N):
    """
    Returns: nested list of scaled weights, rounded to 4 decimals.
    """
    W = torch.tensor(W, dtype=torch.float64)
    scaled_W = W * (1.0/math.sqrt(N))
    return [[round(float(v.item()), 4) for v in row] for row in scaled_W]

def forward_with_scaling(x, weights_list, N, use_scaling):
    """
    Returns: L2 norm of final activation as float, rounded to 4 decimals.
    """
    x = torch.tensor(x, dtype=torch.float64)
    
    if use_scaling:
        for W in weights_list:
            W = torch.tensor(W, dtype=torch.float64)
            x = x + (W * (1.0/math.sqrt(N))) @ x
        out = torch.linalg.norm(x, ord=2)
        return round(float(out.item()), 4)
        
    else:
        for W in weights_list:
            W = torch.tensor(W, dtype=torch.float64)
            x = x + W @ x 
        out = torch.linalg.norm(x, ord=2)
        return round(float(out.item()), 4)
    