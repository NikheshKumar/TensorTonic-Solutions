import numpy as np

def linear_beta_schedule(num_steps: int, beta_start: float, beta_end: float) -> dict:
    """
    Returns a dict with float64 arrays "betas", "alphas" and "alpha_bars", each of shape (num_steps,).
    """

    betas = np.linspace(beta_start, beta_end, num_steps)
    betas = np.asarray(betas, dtype=np.float64)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)

    return {"betas":betas, "alphas":alphas, "alpha_bars":alpha_bars}
    
    
