import numpy as np

def linear_regression(X, y, lr, epochs):
    """
    Returns: tuple (weights, bias)
    """
    
    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.float64)

    n, d = X.shape

    W = np.zeros((d,), np.float64)
    b = 0.0


    for i in range(epochs):
        y_hat = X @ W + b

        grad_W = (2.0/n)*(X.T@(y_hat-y))
        grad_b = (2.0/n)*np.sum(y_hat-y)

        W = W - lr * grad_W
        b = b - lr * grad_b


    return W.tolist(), b.astype(np.float64)
