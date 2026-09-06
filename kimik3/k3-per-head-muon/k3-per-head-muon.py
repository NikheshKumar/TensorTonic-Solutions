import torch

def per_head_muon(parameter: torch.Tensor, gradient: torch.Tensor, previous_momentum: torch.Tensor, num_heads: int, momentum_coefficient: float, learning_rate: float) -> dict[str, torch.Tensor]:
    """
    Returns a dictionary containing the updated parameter, momentum, and orthogonalized update.
    """
    M = momentum_coefficient * previous_momentum + gradient
    M_split = torch.chunk(M, num_heads, dim=0)
    p = []
    
    for M_th in M_split:
        Uh, Eh, Vht = torch.linalg.svd(M_th, full_matrices=False)
        p.append( Uh @ Vht )

    polar = torch.cat(p, dim=0)
    parameter_updated = parameter - learning_rate * polar

    return {"updated_parameter":parameter_updated, "updated_momentum": M, "orthogonalized_update":polar}