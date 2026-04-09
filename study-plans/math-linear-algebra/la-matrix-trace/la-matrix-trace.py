import numpy as np

def matrix_trace(A):
    """
    Returns: float, the trace (sum of diagonal elements) of A.
    """
    A = np.asarray(A, np.float64)
    return np.sum(np.diag(A))