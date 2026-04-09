import numpy as np

def matrix_determinant(A):
    """
    Returns: float, the determinant of square matrix A.
    """
    A = np.asarray(A, np.float64)
    return np.linalg.det(A)