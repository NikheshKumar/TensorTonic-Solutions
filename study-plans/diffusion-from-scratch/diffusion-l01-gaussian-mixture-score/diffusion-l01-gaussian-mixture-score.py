import numpy as np

def gaussian_mixture_score(x: np.ndarray, weights: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """
    Returns the score grad_x log p(x) of the isotropic Gaussian mixture at every row of x, as a float64 array of shape (N, D).
    """

    N, D = x.shape
    
    log_num = np.log(weights) - (D/2)*np.log(2.0*np.pi*variances) - np.sum((x[:, None, :] - means[None, :, :])**2, axis=-1)/(2.0*variances)

    max_val = np.max(log_num, axis=-1, keepdims=True)
    log_den = max_val + np.log(np.sum(np.exp(log_num - max_val), axis=-1, keepdims=True))

    log_r = log_num - log_den
    
    r = np.exp(log_r)

    c = r / variances

    grad = np.sum(c[:, :, None] * (means[None, :, :] - x[:, None, :]), axis=1)

    return grad

    
