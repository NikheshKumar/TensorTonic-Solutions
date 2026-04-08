import numpy as np

def vector_norms(v):
    """
    Returns: float64 array of shape (3,) containing [L1, L2, L-inf] norms.
    """
    v = np.asarray(v, np.float64)

    l1 = np.linalg.norm(v, ord=1)
    l2 = np.linalg.norm(v, ord=2)
    l_inf = np.max(np.abs(v))

    return np.array([l1, l2, l_inf])