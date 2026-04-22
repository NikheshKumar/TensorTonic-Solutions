def log_transform(values):
    """
    Apply the log1p transformation to each value.
    """
    # Write code here
    import numpy as np 

    eps = 1e-6

    values = np.asarray(values, np.float64)

    return np.clip(np.log(1.0+values), eps, None)