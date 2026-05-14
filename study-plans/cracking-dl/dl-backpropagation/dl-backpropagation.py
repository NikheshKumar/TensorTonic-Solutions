import numpy as np

def backprop(X, y, W1, b1, W2, b2):
    """
    Compute gradients for a single-hidden-layer MLP with MSE loss.
    Returns: dict with "dW1", "db1", "dW2", "db2", all rounded to 4 decimals.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    W1 = np.asarray(W1, dtype=np.float64)
    W2 = np.asarray(W2, dtype=np.float64)

    N, D = X.shape

    z1 = X @ W1 + b1 
    a1 = np.maximum(0, z1)
    z2 = a1 @ W2 + b2

    
    dz2 = (2.0 / N) * (z2 - y)
    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0)

    dz1 = (dz2 @ W2.T) * (z1 >= 0.0)
    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0)


    return {"dW1":[np.round(w,4) for w in dW1], "db1":np.round(db1,4), "dW2":[np.round(w,4) for w in dW2], "db2":np.round(db2,4) }
    