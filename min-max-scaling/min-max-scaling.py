def min_max_scaling(data):
    """
    Scale each column of the data matrix to the [0, 1] range.
    """
    # Write code here
    import numpy as np 

    x = np.asarray(data, dtype=np.float64)

    min_x = np.min(x, axis=0)
    max_x = np.max(x, axis=0)

    den = (max_x - min_x)

    den[den==0.0] = 1.0

    x_new = (x-min_x) / den


    return x_new.tolist()