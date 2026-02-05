import numpy as np

def bernoulli_pmf_and_moments(x, p):
    """
    Compute Bernoulli PMF and distribution moments.
    """
    # Write code here

    x = np.asarray(x, int)

    m = p
    var = p*(1-p)

    pmf = np.where(x==0, 1-p, p)
    
    return pmf, m, var