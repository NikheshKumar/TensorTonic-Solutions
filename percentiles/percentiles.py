import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """
    # Write code here
    x = np.asarray(x, np.float64)
    q = np.asarray(q, np.float64)

    x = np.sort(x)

    ans = np.percentile(x, q, method='linear')

    return ans