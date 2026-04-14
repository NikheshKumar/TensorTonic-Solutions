import numpy as np

def low_rank_approximation(A, r):
    """
    Returns: float64 ndarray of shape (m, n), the best rank-r approximation of A.
    """
    A = np.asarray(A, np.float64)

    U, s, Vt = np.linalg.svd(A, full_matrices=False)

    Ur = U[:,:r]
    sr = s[:r]
    Vtr = Vt[:r,:]

    At = Ur @ np.diag(sr) @ Vtr

    return At

    