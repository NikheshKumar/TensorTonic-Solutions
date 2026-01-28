import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """
    # Write code here
    def pmf(n,p,k):
        return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

    PMF = pmf(n,p,k)    

    r = np.arange(0, k + 1)
    #CDF = np.sum(comb(n, r) * (p ** r) * ((1 - p) ** (n - r)))
    CDF = np.sum(pmf(n,p,r))

    return PMF, CDF


    