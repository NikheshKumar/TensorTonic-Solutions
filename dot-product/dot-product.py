import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    # Write code here
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x)!=len(y):
        raise ValueError("Mismatched length of vectors")

    z = sum( x[i]*y[i] for i in range(len(x)))

    return z