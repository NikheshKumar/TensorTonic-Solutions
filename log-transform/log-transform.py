def log_transform(values):
    """
    Apply the log1p transformation to each value.
    """
    # Write code here
    import numpy as np 

    values = np.asarray(values, float)
    y = np.log(1+values)
    eps = 1e-7
    y = np.clip(y, eps, None)

    return y
    