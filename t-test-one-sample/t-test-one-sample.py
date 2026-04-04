import numpy as np

def t_test_one_sample(x, mu0):
    """
    Compute one-sample t-statistic.
    """
    # Write code here
    x = np.asarray(x, float)

    n = len(x)
    s = np.std(x, ddof=1)

    t = (np.mean(x) - mu0)*np.sqrt(n) / s

    return t