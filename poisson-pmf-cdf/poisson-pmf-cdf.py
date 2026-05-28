import numpy as np

def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.
    """
    # Write code here

    def fac(x):
        return np.prod(np.arange(1,x+1))
        
    pmf = [np.exp(-lam) * (lam**i) / fac(i) for i in range(k+1)]

    cdf = np.sum(pmf)


    return pmf[-1], float(cdf)