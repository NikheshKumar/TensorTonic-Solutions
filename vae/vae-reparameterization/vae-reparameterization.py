import numpy as np

def reparameterize(mu: np.ndarray, log_var: np.ndarray) -> np.ndarray:
    """
    Sample from latent distribution using reparameterization trick.
    """
    # Your implementation here
    
    e = np.random.randn(*mu.shape)
    z = mu + np.exp(log_var/2)*e

    return z