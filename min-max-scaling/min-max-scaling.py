def min_max_scaling(data: list) -> list:
    """
    Returns each data column scaled to the range from 0 through 1.
    """
    # Write code here
    import numpy as np 

    x = np.asarray(data, dtype=np.float64)

    if x.size==0:
        return x.tolist()

    min_x = np.min(x, axis=0)
    max_x = np.max(x, axis=0)

    den = (max_x - min_x)

    x_new = np.zeros_like(x)
    np.divide(x - min_x, den, out=x_new, where=(den != 0))

    return x_new.tolist()
    

    