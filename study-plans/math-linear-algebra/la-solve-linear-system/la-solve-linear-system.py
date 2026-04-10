import numpy as np

def solve_linear_system(A, b):
    """
    Returns: float64 array, the solution x to A @ x = b.
    """
    A = np.asarray(A, np.float64)
    b = np.asarray(b, np.float64)

    return np.linalg.solve(A,b)