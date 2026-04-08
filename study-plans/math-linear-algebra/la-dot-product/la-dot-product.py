import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    # Write code here
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)

    return np.dot(x, y)