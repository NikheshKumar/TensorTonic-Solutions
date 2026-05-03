import numpy as np

def bootstrap_mean(x, n_bootstrap=1000, ci=0.95, rng=None):
    """
    Returns: (boot_means, lower, upper)
    """
    # Write code here

    x = np.asarray(x, np.float64)
    n = len(x)

    x_mean = np.zeros(n_bootstrap)

    if rng is None:
        rng = np.random

    for i in range(n_bootstrap):
        random_indices = rng.integers(0,n,size=n)
        x_mean[i] = np.mean(x[random_indices])

    alpha = (1-ci) / 2

    return x_mean, np.quantile(x_mean, alpha), np.quantile(x_mean, 1-alpha)


    
