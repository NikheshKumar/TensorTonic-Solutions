import numpy as np

def outer_product(u, v):
    """
    Returns: float64 matrix of shape (m, n), the outer product u v^T.
    """
    u = np.asarray(u, np.float64)
    v = np.asarray(v, np.float64)

    return np.outer(u, v)