import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here

    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.float64)

    w = np.zeros((X.shape[1],), np.float64)
    b = 0.0

    for i in range(steps):
        
        p = _sigmoid(X@w + b)
        
        grad_w = np.mean(X.T @ (p-y), axis=0)
        grad_b = np.mean(p-y, axis=0)

        w = w - lr * grad_w
        b = b - lr * grad_b

    return w, b