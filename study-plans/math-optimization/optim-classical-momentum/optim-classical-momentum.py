import numpy as np

def momentum_gd(X, y, lr, beta, n_epochs):
    """
    Returns: tuple of (vanilla_losses, momentum_losses), each a list of MSE values
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    ans = []

    N, D = X.shape

    vanilla_losses = []
    momentum_losses = []

    w = np.zeros((D,), dtype=np.float64)
    w_vanilla = np.zeros((D,), dtype=np.float64)
    w_mom = np.zeros((D,), dtype=np.float64)
    v = np.zeros((D,), dtype=np.float64)
    

    for i in range(n_epochs):

        vanilla_grad = 2.0 * X.T @ (X @ w_vanilla - y) / N
        loss_vanilla = np.mean((X@w_vanilla - y)**2)
        w_vanilla = w_vanilla - lr * vanilla_grad


        mom_grad = 2.0 * X.T @ (X @ w_mom - y) / N
        v = beta * v + mom_grad
        loss_mom = np.mean((X@w_mom - y)**2)
        w_mom = w_mom - lr * v

        
        vanilla_losses.append(loss_vanilla)
        momentum_losses.append(loss_mom)


    return (vanilla_losses, momentum_losses)