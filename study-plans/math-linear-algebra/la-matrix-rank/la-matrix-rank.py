import numpy as np

def matrix_rank(A):
    """
    Returns: int, the rank of matrix A.
    """
    A = np.asarray(A, np.float64)

    return np.linalg.matrix_rank(A)