import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """
    # Write code here
    def pmf(n,p,k):
      v1 = (p**k)
      v2 = (1-p)**(n-k)
      return comb(n,k) * v1 * v2

    PMF = pmf(n,p,k)

    r = np.arange(0,k+1,1)
    cdf = np.sum(pmf(n,p,r))


    return float(PMF), float(cdf)