import numpy as np

def cosine_restarts(X: list, y: list, eta_max: float, eta_min: float, T_0: int, T_mult: int, total_epochs: int) -> dict:
    """
    Returns the cosine-restart schedule and pre-update losses.
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    N, D = X.shape

    w = np.zeros((D,), dtype=np.float64)

    lr_schedule = []
    losses = []

    T_i = T_0
    T_curr = 0.0
    
    for i in range(total_epochs):

        alpha = eta_min + (eta_max - eta_min)*(1+ np.cos(np.pi * T_curr/T_i))/2.0
        lr_schedule.append(alpha)

        loss = np.mean((X@w-y)**2)
        losses.append(loss)

        w -= alpha * (2.0/N) * X.T @ (X@w-y)
        
        T_curr += 1
        if T_curr == T_i:
            T_i = T_i * T_mult
            T_curr = 0



    return {"lr_schedule":lr_schedule, "losses":losses}
        