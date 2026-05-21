import numpy as np

def make_diagonal(v):
    """
    Returns: (n, n) NumPy array with v on the main diagonal
    """
    # Write code here
    v = np.asarray(v, np.float64).ravel()
    n = len(v)
    A = np.diag(v)
    return A