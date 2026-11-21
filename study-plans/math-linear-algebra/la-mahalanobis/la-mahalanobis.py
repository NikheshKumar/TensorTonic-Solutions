import numpy as np

def mahalanobis_distance(x: list, mean: list, cov: list) -> float:
    """
    Returns the Mahalanobis distance as a Python float.
    """
    x = np.asarray(x, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)

    S = np.linalg.solve(cov, x-mean)
    d = np.sqrt((x-mean).T @ S)

    return float(d)

    