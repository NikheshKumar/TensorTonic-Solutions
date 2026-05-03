import numpy as np

def bootstrap_mean(x, n_bootstrap=1000, ci=0.95, rng=None):
    """
    Returns: (boot_means, lower, upper)
    """
    # Write code here

    x = np.asarray(x, np.float64)
    n = len(x)


    if rng is None:
        rng = np.random

    random_indices = rng.integers(0,n,size=(n_bootstrap,n))

    x_mean = np.mean(x[random_indices], axis=1)

    alpha = (1-ci) / 2

    return x_mean, np.quantile(x_mean, alpha), np.quantile(x_mean, 1-alpha)


    
