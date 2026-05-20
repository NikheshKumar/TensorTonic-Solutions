import numpy as np

def eigendecompose(A):
    """
    Returns: tuple (eigenvalues, eigenvectors), sorted by descending magnitude.
    """
    A = np.asarray(A, np.float64)

    vals, vecs = np.linalg.eig(A)
    idx = np.argsort(np.abs(vals))[::-1]

    vals = vals[idx]
    vecs = vecs[:, idx]

    for v in vecs:
        if np.linalg.norm(v) > 1e-8:
            v = v / np.linalg.norm(v)

    return vals, vecs

    