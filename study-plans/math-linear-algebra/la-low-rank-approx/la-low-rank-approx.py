import numpy as np

def low_rank_approximation(A, r):
    """
    Returns: float64 ndarray of shape (m, n), the best rank-r approximation of A.
    """
    A = np.asarray(A, np.float64)

    U, s, Vt = np.linalg.svd(A, full_matrices=True)

    U = U[:,:r]
    s = s[:r]
    Vt = Vt[:r,:]

    At = U @ np.diag(s) @ Vt

    return At

    