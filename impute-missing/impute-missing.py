import numpy as np

def impute_missing(X: list, strategy: str = "mean") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as X.
    """
    # Write code here
    X = np.array(X, dtype=np.float64)

    n = X.shape[-1]
    mask = np.isnan(X)

    if strategy == "mean":
        m = np.nanmean(X, axis=0)

    if strategy == "median":
        m = np.nanmedian(X, axis=0)
       

    m = np.where(np.isnan(m), 0.0, m)
    X[mask] = np.broadcast_to(m, X.shape)[mask]

    return X
            