import numpy as np

def geometric_pmf_mean(k, p):
    """
    Compute Geometric PMF and Mean.
    """
    # Write code here
    k = np.asarray(k, int)
    pmf = ((1-p)**(k-1)) * p

    e = 1.0/p

    return pmf, float(e)