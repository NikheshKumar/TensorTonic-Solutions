import numpy as np

def projection_matrix(A: list) -> np.ndarray:
    """
    Returns the float64 projector onto the column space of A.
    """
    A = np.asarray(A, dtype=np.float64)

    G = A.T @ A

    P = np.linalg.solve(G, A.T)

    P = A @ P

    return P
    