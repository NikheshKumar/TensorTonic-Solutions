import numpy as np

def least_squares(A: list, b: list) -> np.ndarray:
    """
    Returns the float64 least-squares solution vector.
    """
    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    
    Q, R = np.linalg.qr(A, mode="reduced")

    x = np.linalg.solve(R, Q.transpose()@b)

    return x