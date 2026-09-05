import numpy as np

def ddpm_sample(x_T: list, betas: list[float], epsilon_preds: list, z_values: list) -> list:
    """
    Returns the final denoised sample rounded to four decimals.
    """
    x_T = np.array(x_T, dtype=np.float64)
    
    if x_T.ndim == 1:
        x_T = x_T[np.newaxis, :]     
        
    betas = np.array(betas, dtype=np.float64)
    epsilon_preds = np.array(epsilon_preds, dtype=np.float64)
    z_values = np.array(z_values, dtype=np.float64)

    T = len(betas)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)


    for t in range(T, 0, -1):

        mean = (1.0 / np.sqrt(alphas[t-1])) * (
            x_T- betas[t-1] * epsilon_preds[T-t] / np.sqrt(1.0 - alpha_bars[t-1])
        )
        if t > 1:
            x_T = mean + np.sqrt(betas[t-1]) * z_values[T-t]
        else:
            x_T = mean

    return np.round(x_T, 4).tolist()