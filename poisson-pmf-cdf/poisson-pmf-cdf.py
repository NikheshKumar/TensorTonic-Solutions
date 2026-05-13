import numpy as np

def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.
    """
    # Write code here
    def pmf(l, i):
        if i == 0:
            return np.exp(-l)
        return (np.exp(-l) * (l**i)) / np.prod(np.arange(1, i + 1))


    arr = [pmf(lam,i) for i in range(k+1)]
    cdf = np.sum(arr)


    return pmf(lam, k).astype(np.float64), float(cdf)