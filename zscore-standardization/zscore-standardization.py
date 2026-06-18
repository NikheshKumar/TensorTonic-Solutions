import numpy as np

def zscore_standardize(X, axis=0, eps=1e-12):
    """
    Standardize X: (X - mean)/std. If 2D and axis=0, per column.
    Return np.ndarray (float).
    """
    # Write code here
    X = np.asarray(X, dtype=np.float64)

    X_new = (X-X.mean(axis=axis, keepdims=True))/( X.std(axis=axis, keepdims=True) + eps)


    return X_new