import numpy as np

def logistic_regression(X, y, lr=0.01, n_iters=1000):
    """
    Returns:
        tuple: (weights, bias) where weights is a list and bias is a float
    """
    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.float64)

    n, d = X.shape

    W = np.zeros((d,), dtype=np.float64)
    b = 0.0

    def _sigmoid(x):
        return np.exp(x) / (1+np.exp(x))

    for i in range(n_iters):

        y_hat = _sigmoid(X@W + b)

        W = W - lr * (1.0/n) * (X.T @ (y_hat - y))
        b = b - lr * (1.0/n) * np.sum(y_hat - y)


    return W.tolist(), b.astype(np.float64)
        
