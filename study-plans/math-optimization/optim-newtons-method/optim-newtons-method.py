import numpy as np

def newtons_method(X: list, y: list, lr_gd: float, n_iters: int) -> dict:
    """
    Returns gradient-descent and Newton MSE curves in a dictionary.
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    N, D = X.shape

    X_aug = np.column_stack((X, np.ones(N)))

    w_gd = np.zeros((D+1,), dtype=np.float64)
    w_new = np.zeros((D+1,), dtype=np.float64)

    gd_mse = []
    newton_mse = []

    H = (2.0/N)*(X_aug.T @ X_aug)

    gd_mse.append(float(np.mean((X_aug@w_gd-y)**2)))
    newton_mse.append(float(np.mean((X_aug@w_new-y)**2)))

    for i in range(n_iters):

        grad_gd = (2.0 / N) * (X_aug.T @ (X_aug@w_gd - y))
        grad_new = (2.0 / N) * (X_aug.T @ (X_aug@w_new - y) ) 
        
        w_gd -= lr_gd * grad_gd
        w_new -= np.linalg.solve(H, grad_new)

        l_gd = float(np.mean((X_aug@w_gd-y)**2))
        l_new = float(np.mean((X_aug@w_new-y)**2))

        gd_mse.append(l_gd)
        newton_mse.append(l_new)


    return {"gd_losses":gd_mse, "newton_losses":newton_mse}
        
        
        