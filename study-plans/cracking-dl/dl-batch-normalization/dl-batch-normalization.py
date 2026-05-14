import numpy as np

def batch_norm(X, gamma, beta, running_mean, running_var, mode):
    """
    Returns: dict with keys "output", "running_mean", "running_var"
    """
    X = np.asarray(X, np.float64)
    gamma = np.asarray(gamma, np.float64)
    beta = np.asarray(beta, np.float64)
    running_mean = np.asarray(running_mean, np.float64)
    running_var = np.asarray(running_var, np.float64)

    eps = 1e-5
    m = 0.1

    if mode=='train':
        mu = np.mean(X, axis=0)
        var = np.var(X, axis=0)

        X_new = (X-mu) / np.sqrt(var + eps)
        y = gamma * X_new + beta
        
        running_mean = (1-m)*(running_mean) + m*mu
        running_var = (1 - m) * running_var + m * var


    else:
        X_new = (X-running_mean) / np.sqrt(running_var + eps)
        y = gamma * X_new + beta

    return {"output":[[round(float(y_row),4) for y_row in row] for row in y], "running_mean":[round(float(rm),4) for rm in running_mean], "running_var":[round(float(rv),4) for rv in running_var]}
