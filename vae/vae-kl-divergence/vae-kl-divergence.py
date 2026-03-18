import numpy as np

def kl_divergence(mu: np.ndarray, log_var: np.ndarray) -> float:
    """
    Compute KL divergence between q(z|x) and N(0, I).
    """
    # Your implementation here
    var = np.exp(log_var)
  
    d_kl_per_sample = -np.sum(1+log_var-np.square(mu)-var,axis=1) / 2

    d_kl = np.mean(d_kl_per_sample)

    return float(d_kl)

  