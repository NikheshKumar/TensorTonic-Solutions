import torch

def critical_batch_size(batch_sizes: torch.Tensor, steps_to_target: torch.Tensor, reached_mask: torch.Tensor) -> dict:
    """
    Returns a dict: examples (list), min_steps, min_examples, critical_batch_size.
    """
    E_full = batch_sizes.to(torch.float64) * steps_to_target.to(torch.float64)

    E_reached = E_full[reached_mask]
    S_reached = steps_to_target[reached_mask]
    
    Smin = torch.min(S_reached)
    Emin = torch.min(E_reached)

    B_crit = Emin / Smin

    examples = [float(e) if bool(mask) else None for e, mask in zip(E_full.tolist(), reached_mask.tolist())]

    return {"examples":examples,"min_steps":float(Smin.item()), "min_examples":float(Emin.item()), "critical_batch_size":float(B_crit.item())}
