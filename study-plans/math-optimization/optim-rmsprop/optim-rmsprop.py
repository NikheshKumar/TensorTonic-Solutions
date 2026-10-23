import numpy as np

def rmsprop(X: list, y: list, lr: float, decay: float, n_epochs: int) -> dict:
    """
    Returns MSE losses and effective learning rates in a dictionary.
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    N, D = X.shape

    G = np.zeros((D,), dtype=np.float64)
    w = np.zeros((D,), dtype=np.float64)

    L, E = [], []

    eps = 1e-8

    for i in range(n_epochs):
        l = np.mean((X@w-y)**2)
        g = (2.0/N)*X.T @ (X@w-y)
        G = decay * G + (1-decay)*(g**2)
        elr = lr/np.sqrt(G + eps)
        w = w - g * elr 

        L.append(l)
        E.append(elr)


    return {"losses":L, "effective_lrs":E}
        

        