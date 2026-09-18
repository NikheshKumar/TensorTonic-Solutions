import numpy as np

def pearson_correlation(X: list) -> np.ndarray:
    """
    Returns the correlation matrix as a NumPy array.
    """
    # Write code here
    X = np.array(X, dtype=np.float64)

    if X.ndim!=2:
        return None

    N, D = X.shape
    if N<2:
        return None

    X_c = X - np.mean(X, axis=0, keepdims=True)

    cov = X_c.T @ X_c / (N-1)

    std_dev = np.std(X, axis=0, keepdims=True, ddof=1)

    den = np.outer(std_dev, std_dev)

    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.divide(cov, den, out=np.full((D, D), np.nan), where=den != 0)

    mask = std_dev != 0

    corr[np.diag_indices_from(corr)] = np.where(mask, 1.0, np.nan)

    return corr