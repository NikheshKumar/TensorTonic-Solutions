import numpy as np
import math

def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.
    """
    # Write code here
    indices = np.arange(k+1)
    arr = np.ones(k+1)

    if k<0:
      return 0.0, 0.0
    else:
      arr[1:] = np.cumprod(np.arange(1, k + 1))

    arr = np.exp(-lam) * (lam**indices) / arr

    pmf = arr[-1]

    cdf = np.sum(arr)

    return float(pmf), float(cdf)