import numpy as np

def reverse_step(x_t: list, t: int, epsilon_pred: list, betas: list[float], z: list = None) -> list:
    """
    Returns x at timestep t - 1, rounded to four decimals.
    """
    
    x_t = np.array(x_t, dtype=np.float64)
    betas = np.array(betas, dtype=np.float64)
    epsilon_pred = np.array(epsilon_pred, dtype=np.float64)
    
    if z is not None and len(z) > 0:
        z = np.array(z, dtype=np.float64)
    else:
        z = np.zeros_like(x_t)

    
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas)


    m = (1.0 / np.sqrt(alphas[t-1])) * (x_t - (betas[t-1] / np.sqrt(1.0 - alpha_bar[t-1])) * epsilon_pred)

    x_prev = m

    if t > 1:
        x_prev += np.sqrt(betas[t-1])* z


    return np.round(x_prev, 4).tolist()