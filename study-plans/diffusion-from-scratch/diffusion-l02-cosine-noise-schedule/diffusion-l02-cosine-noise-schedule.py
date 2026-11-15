import numpy as np

def cosine_noise_schedule(num_steps: int, s: float, max_beta: float) -> dict:
    """
    Returns a dict with float64 arrays "betas" and "alpha_bars", each of shape (num_steps,).
    """

    def f(u):
        num = (u/num_steps + s)*np.pi
        den = 2.0*(1.0+s)
        z = np.cos(num/den)
        return z**2
        
    betas = [min(1-f(t+1)/f(t), max_beta) for t in range(num_steps)]
    betas = np.asarray(betas, dtype=np.float64)

    alpha_bars = np.cumprod(1-betas)

    return {"betas":betas, "alpha_bars":alpha_bars}

    
