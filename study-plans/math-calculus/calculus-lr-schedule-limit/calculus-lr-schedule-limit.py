import numpy as np

def lr_schedule_analysis(alpha_0, k):
    """
    Returns: dict with 'limit' (float), 'sum_diverges' (bool), 'sum_sq_converges' (bool)
    """
    
    limit = 0.0 if k>0.0 else np.float64(alpha_0)

    sum_diverges = bool(alpha_0 > 0.0)

    sum_sq_converges = bool((k>0.0) or (alpha_0==0.0))

    return {"limit":limit, "sum_diverges":sum_diverges, "sum_sq_converges":sum_sq_converges}