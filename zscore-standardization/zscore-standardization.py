import numpy as np

def zscore_standardize(X, axis=0, eps=1e-12):
    """
    Standardize X: (X - mean)/std. If 2D and axis=0, per column.
    Return np.ndarray (float).
    """
    # Write code here
    X = np.asarray(X, float)
    m = np.mean(X, axis=axis, keepdims=True)
    std_dev = np.std(X, axis=axis, keepdims=True)

    Z = (X - m) / (std_dev + eps)
    return Z.tolist()