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

        y_hat = np.dot(X,W) + b

        grad_W = (2.0/n) * np.dot(X.T, y_hat-y) + alpha*np.sign(W)
        grad_b = (2.0/n) * np.sum(y_hat-y)

        W = W - lr*grad_W
        b = b - lr*grad_b


    return W.tolist(), float(b)