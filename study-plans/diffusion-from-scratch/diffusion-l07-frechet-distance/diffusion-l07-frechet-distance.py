import numpy as np

def frechet_distance(features_a: np.ndarray, features_b: np.ndarray) -> float:
    """
    Returns the squared Frechet distance between Gaussian fits of the two feature sets as a Python float.
    """
    A = np.asarray(features_a, dtype=np.float64)
    B = np.asarray(features_b, dtype=np.float64)
    
    if A.ndim == 1:
        A = A[:, None]
    if B.ndim == 1:
        B = B[:, None]

    mean_a = np.mean(A, axis=0)
    mean_b = np.mean(B, axis=0)
    
        
    cov_a = np.atleast_2d(np.cov(A, rowvar=False))
    cov_b = np.atleast_2d(np.cov(B, rowvar=False))
    tr_a = float(np.trace(cov_a))
    tr_b = float(np.trace(cov_b))

    tr_ab = np.trace(cov_a @ cov_b)

    d_sq = (mean_a - mean_b).T @ (mean_a - mean_b)

    evals = np.linalg.eigvals(cov_a @ cov_b)
    sqrt_evals = np.sqrt(np.maximum(evals.real, 0.0))

    d_sq += tr_a + tr_b - 2.0 * np.sum(sqrt_evals)

    return float(d_sq)

