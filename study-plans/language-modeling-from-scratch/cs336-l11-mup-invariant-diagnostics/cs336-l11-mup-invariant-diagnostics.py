import torch

def mup_invariant_diagnostics(
    widths: torch.Tensor, activation_rms: torch.Tensor,
    activation_change_rms: torch.Tensor, slope_tolerance: int | float,
) -> dict:
    """
    Returns a dict of slopes, intercepts, ratios (floats) and invariant decisions (bools).
    """
    widths = widths.to(torch.float64)
    activation_rms = activation_rms.to(torch.float64)
    activation_change_rms = activation_change_rms.to(torch.float64)
    
    R_act = torch.max(activation_rms)/torch.min(activation_rms)

    R_change = torch.max(activation_change_rms)/torch.min(activation_change_rms)

    log_y1 = torch.log(activation_rms)
    log_y2 = torch.log(activation_change_rms)

    log_w = torch.log(widths)

    A = torch.stack([log_w, torch.ones_like(log_w)], dim=1)
    b1 = log_y1.unsqueeze(-1)
    b2 = log_y2.unsqueeze(-1)

    sol1 = torch.linalg.lstsq(A,b1).solution
    sol2 = torch.linalg.lstsq(A,b2).solution

    beta1 = sol1[0,0].item()
    beta2 = sol2[0,0].item()

    a1 = sol1[1,0].item()
    a2 = sol2[1,0].item()

    act_inv = bool(abs(beta1) <= slope_tolerance)

    ch_inv = bool(abs(beta2) <= slope_tolerance)


    return {"activation_slope":beta1,"activation_intercept":a1,"change_slope":beta2,"change_intercept":a2,"activation_ratio":float(R_act),"change_ratio":float(R_change),"activation_invariant":act_inv,"change_invariant":ch_inv}
