import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    # Write code here
    x = np.asarray(x, np.float64)

    return np.var(x,ddof=1).astype(np.float64), np.std(x, ddof=1).astype(np.float64)