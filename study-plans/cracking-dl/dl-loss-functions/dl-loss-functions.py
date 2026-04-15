import numpy as np

def loss_functions(y_true, y_pred, loss_type):
    """
    Returns: Loss value as a float, rounded to 4 decimal places.
    """
    y_pred = np.asarray(y_pred, np.float64)
    y_true = np.asarray(y_true, np.float64)
    y_true_new = np.asarray(y_true, int)

    eps = 1e-15

    if loss_type == "mse":
        out = np.mean((y_pred-y_true)**2)
        
    if loss_type == "bce":
        y_pred = np.clip(y_pred, eps, 1 - eps)
        l = y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred)
        out = -np.mean(l)
        
    if loss_type == "cce":
        
        exps = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
        p = exps / np.sum(exps, axis=1, keepdims=True)
        
        num_samples = len(y_true)
        indices = y_true_new.astype(int)

        log_p = -np.log(p[np.arange(num_samples), indices])
        out = np.mean(log_p)
        
    if loss_type == "hinge":
        out = np.mean(np.maximum(0,1-y_pred*y_true))

    return (np.round(out, 4)).astype(np.float64)
        
    