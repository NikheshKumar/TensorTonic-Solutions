import numpy as np

def linear_beta_schedule(T: int, beta_1: float = 0.0001, beta_T: float = 0.02) -> np.ndarray:
    """
    Linear noise schedule from beta_1 to beta_T.
    """
    # YOUR CODE HERE
    beta = np.linspace(beta_1, beta_T, T)
    return beta

def cosine_alpha_bar_schedule(T: int, s: float = 0.008) -> np.ndarray:
    """
    Cosine schedule for alpha_bar (cumulative signal retention).
    """
    # YOUR CODE HERE
    t = np.linspace(0,T,T+1)
    f_t = ft = np.cos(((t / T + s) / (1 + s)) * (np.pi / 2))**2
    alpha_bars = f_t / f_t[0]
    return alpha_bars
    

def alpha_bar_to_betas(alpha_bars: np.ndarray) -> np.ndarray:
    """
    Convert alpha_bar schedule to beta schedule.
    """
    # YOUR CODE HERE
    alpha_bars = np.insert(alpha_bars, 0, 1.0)
    alphas = alpha_bars[1:] / alpha_bars[:-1]
    betas = (1.0 - alphas)
    return np.clip(betas, a_min=0, a_max=0.999)
    
