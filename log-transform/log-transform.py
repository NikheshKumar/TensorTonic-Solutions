def log_transform(values):
    """
    Apply the log1p transformation to each value.
    """
    # Write code here
    import numpy as np 
    import math

    values = np.asarray(values, float)
    y = np.log1p(values)
    eps = 1e-7
    y = np.clip(y, eps, None)

    return y
    