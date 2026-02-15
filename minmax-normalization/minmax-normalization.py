import numpy as np

def minmax_scale(X, axis=0, eps=1e-12):
    """
    Scale X to [0,1]. If 2D and axis=0 (default), scale per column.
    Return np.ndarray (float).
    """
    # Write code here
    X = np.asarray(X)

    min_x = np.min(X, axis=axis, keepdims=True)
    max_x = np.max(X, axis=axis, keepdims=True)

    denominator = (max_x - min_x)

    X_new = (X - min_x) / np.maximum(denominator, eps)

    return X_new.tolist()

    


  