import numpy as np

def get_alpha_bar(betas: list[float]) -> list[float]:
    """
    Returns the cumulative alpha-bar values rounded to six decimals.
    """
    betas = np.array(betas, dtype=np.float64)
    alpha_bar = np.cumprod(1.0 - betas)
    return alpha_bar

def forward_diffusion(x_0: list, t: int, betas: list[float], epsilon: list) -> list:
    """
    Returns x_t with the same nested shape as x_0.
    """
    x_0  = np.array(x_0, dtype=np.float64)
    betas = np.array(betas, dtype=np.float64)
    epsilon = np.array(epsilon, dtype=np.float64)

    alpha_bar = get_alpha_bar(betas)

    x_t = np.sqrt(alpha_bar[t-1]) * x_0 + np.sqrt(1.0- alpha_bar[t-1]) * epsilon

    return x_t
    