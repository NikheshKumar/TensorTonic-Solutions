import numpy as np

def svd(A):
    """
    Returns: tuple (U, s, Vt) where A = U @ diag(s) @ Vt.
    """
    A = np.asarray(A, np.float64)

    U,s,Vt = np.linalg.svd(A, full_matrices=False)

    return (U,s,Vt)