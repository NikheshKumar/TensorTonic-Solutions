import numpy as np

def adam(X: list, y: list, lr: float, beta1: float, beta2: float, n_epochs: int) -> dict:
    """
    Returns pre-update MSE losses and final weights in a dictionary.
    """
    X = np.array(X, dtype=np.float64)
    y =np.array(y, dtype=np.float64)

    N, D = X.shape

    m = np.zeros((D,), dtype=np.float64)
    v = np.zeros((D,), dtype=np.float64)
    g = np.zeros((D,), dtype=np.float64)
    w = np.zeros((D,), dtype=np.float64)

    eps = 1e-8

    L, Weights = [], []

    for i in range(1,n_epochs+1):
        
        l = np.mean((X@w-y)**2)
        g = (2.0/N)*X.T @ (X@w-y)
        m = beta1*m + (1-beta1)*g
        v = beta2*v + (1-beta2)*(g**2)
        m_hat = m/(1-beta1**i)
        v_hat = v/(1-beta2**i)
        w = w - lr * m_hat/(np.sqrt(v_hat)+eps)

        L.append(l)

    return {"losses":L, "final_weights":w.tolist()}
        
        
    

    