import numpy as np

def bernoulli_pmf_and_moments(x, p):
    """
    Compute Bernoulli PMF and distribution moments.
    """
    # Write code here
    x = np.asarray(x, np.float64)
    
    pmf = np.where(x==1, p, 1-p)

    mean = p

    var = p*(1-p)

    return pmf, mean, var