import torch
import torch.nn.functional as F

def sgns_loss(center_vec: torch.Tensor, pos_vec: torch.Tensor, neg_vecs: torch.Tensor) -> torch.Tensor:
    """
    Returns a scalar torch.Tensor: the SGNS loss.
    """
    # YOUR CODE HERE
    Lp = F.softplus(-torch.sum(center_vec * pos_vec, dim=-1))
    Ln = torch.sum(F.softplus(torch.sum(center_vec * neg_vecs, dim=-1)), dim=-1)

    return Lp + Ln
