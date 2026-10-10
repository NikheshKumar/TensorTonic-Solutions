import numpy as np

def svm_hinge_sgd(X: list, y: list, lr: float, lam: float, n_epochs: int) -> dict:
    """
    Returns fitted parameters and training predictions.
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    w = np.zeros((X.shape[1],), dtype=np.float64)
    b = 0.0 

    for _ in range(n_epochs):

        for i in range(X.shape[0]):
            xi = X[i]
            yi = y[i]

            margin = yi * (w @ xi + b)

            if margin<1:
                w -= lr * (lam * w - yi * xi)
                b += lr * yi
            else:
                w -= lr * lam * w
        

    scores = X @ w + b
    y_preds = np.where(scores>0.0, 1, -1)

    return {"bias":b, "predictions":y_preds.tolist(), "weights":w.tolist()}
