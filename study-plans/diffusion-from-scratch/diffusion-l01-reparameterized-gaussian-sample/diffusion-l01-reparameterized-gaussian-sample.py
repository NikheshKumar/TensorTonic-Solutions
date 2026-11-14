import numpy as np

def reparameterized_sample(mean: np.ndarray, log_variance: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """
    Returns the reparameterized samples as a float64 array with the broadcast shape of the inputs.
    """
    z = mean + np.exp(0.5 * log_variance) * noise

    return z
