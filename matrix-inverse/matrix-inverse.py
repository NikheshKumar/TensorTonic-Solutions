import numpy as np

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I
    """
    # Write code here
    A = np.asarray(A, float)
    n,d = A.shape

    if A.ndim != 2 or n != d:
        return None

    eps = 1e-12
    det_A = np.linalg.det(A)

    if abs(det_A) < eps:
        return None

    A_inv = np.linalg.inv(A) 
    return A_inv  


        



