import numpy as np

def adagrad(X: list, y: list, lr: float, n_epochs: int) -> dict:
    """
    Returns MSE losses and effective learning rates in a dictionary.
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    N, D = X.shape

    G = np.zeros((D,), dtype=np.float64)
    w = np.zeros((D,), dtype=np.float64)

    eps = 1e-8
    L, E = [], []
    
    for i in range(n_epochs):
        l = np.mean((X@w-y)**2)
        g = (2.0/N) * X.T @ (X@w - y)
        G = G + g**2
        eff = lr / (np.sqrt(G + eps))
        w = w - g * eff

        
        L.append(l)
        E.append(eff)



    return {"losses":L, "effective_lrs":E}
        
        