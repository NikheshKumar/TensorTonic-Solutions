def lasso_regression(X, y, lr, epochs, alpha):
    """
    Perform Lasso Regression using gradient descent with L1 subgradient.
    Returns: tuple of (weights_list, bias_float)
    """
    import numpy as np 

    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.float64)

    n, d = X.shape

    W = np.zeros((d,), np.float64)
    b = 0.0

    for i in range(epochs):

        y_hat = X @ W + b
        error = y_hat - y
        grad_W = (2.0 / n) * (X.T @ error) + alpha * np.sign(W)
        grad_b = (2.0 / n) * np.sum(error)
        W -= lr * grad_W
        b -= lr * grad_b


    return W.tolist(), float(b)