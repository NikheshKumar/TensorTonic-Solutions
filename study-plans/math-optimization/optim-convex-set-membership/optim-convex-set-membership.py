import numpy as np

def convex_set_membership(A, b, x):
    """
    Returns: dict with 'in_set' (bool) and 'max_violation' (float, rounded to 6 decimals)
    """

    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    max_violation = np.max(A@x - b, axis=0)

    in_set = bool(max_violation <= 1e-6)

    return {"in_set":in_set, "max_violation":np.round(max_violation,6)}
