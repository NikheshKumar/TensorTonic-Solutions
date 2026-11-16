import numpy as np

def rescale_zero_terminal_snr(betas: np.ndarray) -> dict:
    """
    Returns a dict with float64 arrays "betas", "alpha_bars" and "snr" of the rescaled schedule, each of shape (T,).
    """
    T = betas.shape[0]
    
    alpha_bars = np.cumprod(1.0 - betas)

    a = np.sqrt(alpha_bars)

    a_tilde = (a - a[-1]) * a[0]/(a[0] - a[-1])

    alpha_bars_tilde = a_tilde ** 2

    snr = alpha_bars_tilde / (1.0 - alpha_bars_tilde)

    betas_tilde = [1.0 - alpha_bars_tilde[0]]

    for i in range(1, T):
        betas_tilde.append(1.0 - alpha_bars_tilde[i]/alpha_bars_tilde[i-1])

    betas_tilde = np.asarray(betas_tilde, dtype=np.float64)

    return {"betas":betas_tilde, "alpha_bars":alpha_bars_tilde, "snr":snr} 
