import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    # Write code here
    x = np.asarray(x, float)

    m = np.mean(x)
    n = len(x)

    sam_var = ( np.sum((x-m)**2) ) / (n-1)
    std_dev = np.sqrt(sam_var)

    return sam_var, std_dev