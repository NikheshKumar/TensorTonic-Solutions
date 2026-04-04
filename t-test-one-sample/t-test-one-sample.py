import numpy as np

def t_test_one_sample(x, mu0):
    """
    Compute one-sample t-statistic.
    """
    # Write code here
    x = np.asarray(x, float)

    n = len(x)
    s = np.std(x, ddof=1)
    mu_x = np.mean(x)

    if n <= 1:
        return 0.0

    if s == 0.0:
        if mu_x == mu0:
            return 0.0  
        else:
            return float('inf')

    t = (mu_x- mu0)*np.sqrt(n) / s

    return float(t)