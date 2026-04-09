import numpy as np

def matrix_multiply(A, B):
    """
    Returns: 2-D float64 array, the matrix product A @ B.
    """
    return np.asarray(A, np.float64) @ np.asarray(B, np.float64)