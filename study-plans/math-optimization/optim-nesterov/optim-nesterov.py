import numpy as np

def nesterov_momentum(X: list, y: list, lr: float, beta: float, n_epochs: int) -> dict:
    """
    Returns classical and Nesterov MSE loss curves in a dictionary.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    N, D = len(X), len(X[0])

    wc = np.zeros(D, dtype=np.float64)
    wn = np.zeros(D, dtype=np.float64)
    vc = np.zeros(D, dtype=np.float64)
    vn = np.zeros(D, dtype=np.float64)


    loss_c = []
    loss_n = []

    for _ in range(n_epochs):
        
        # classical
        
        y_pred_c = X @ wc 
        err_c = y_pred_c - y
        
        loss_c.append(float(np.mean(err_c ** 2)))

        grad_wc = (2.0 / N) * (X.T @ err_c)
        vc  = beta * vc  + grad_wc
        wc = wc - lr * vc
        
        # nesterov
        
        y_pred_n = X @ wn 
        err_n = y_pred_n - y
        loss_n.append(float(np.mean(err_n ** 2)))

        w_lookahead = wn - lr * beta * vn

        y_pred_lookahead = X @ w_lookahead 
        err_lookahead = y_pred_lookahead - y

        grad_wn = (2.0 / N) * (X.T @ err_lookahead)
        vn  = beta * vn  + grad_wn
        wn = wn - lr * vn
    

    return {"classical_losses": loss_c, "nesterov_losses": loss_n}
        