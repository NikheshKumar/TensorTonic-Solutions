import numpy as np

def hadamard_product(A, B):
    """
    Returns: ndarray, the element-wise product A * B.
    """
    A = np.asarray(A, np.float64)
    B = np.asarray(B, np.float64)

    return A*B