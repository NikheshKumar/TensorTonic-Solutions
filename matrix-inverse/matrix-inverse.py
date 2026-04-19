import numpy as np

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I
    """
    # Write code here
    
    A = np.asarray(A, np.float64)
    n,d = A.shape

    if n!=d or A.ndim!=2:
        return None

    eps=1e-12
    if abs(np.linalg.det(A)) < eps:
        return None
        
    return np.linalg.inv(A)
