import numpy as np

def softmax_regression(X, y, n_classes, lr=0.01, n_iters=1000):
    """
    Returns: tuple (weights, bias) where weights is a 2D list (d x K) and bias is a list of length K
    """
    X = np.asarray(X, np.float64)
    y = np.asarray(y, int)

    n, d = X.shape

    W = np.zeros((d,n_classes), np.float64)
    b = np.zeros((n_classes,), np.float64)

    Y_onehot = np.zeros((n, n_classes), np.float64)
    Y_onehot[np.arange(n), y] = 1

    for i in range(n_iters):

        z = X@W + b
        new_z = z-np.max(z, axis=1, keepdims=True)
        P = np.exp(new_z) / np.sum(np.exp(new_z), axis=1, keepdims=True)

        grad_W = (1.0/n)*np.dot(X.T, (P - Y_onehot))
        grad_b = (1.0/n)*np.dot(np.ones(n).T, P - Y_onehot)

        W = W - lr*grad_W
        b = b - lr*grad_b


    return W.tolist(), b.tolist()
