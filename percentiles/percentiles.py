import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """
    # Write code here
    x = np.asarray(x, np.float64)

    ans = np.percentile(x, q, method='linear')

    return np.sort(ans)