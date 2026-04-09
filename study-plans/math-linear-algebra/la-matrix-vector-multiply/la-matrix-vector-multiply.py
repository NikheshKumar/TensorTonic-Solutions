import numpy as np

def matrix_vector_multiply(A, x):
    """
    Returns: 1-D float64 array, the product A @ x.
    """
    A = np.asarray(A, np.float64)
    x = np.asarray(x, np.float64)

    return A@x