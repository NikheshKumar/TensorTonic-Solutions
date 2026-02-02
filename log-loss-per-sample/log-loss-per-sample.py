import math

def log_loss(y_true, y_pred, eps=1e-15):
    """
    Compute per-sample log loss.
    """
    # Write code here

    import numpy as np

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    p = np.clip(y_pred, eps, 1 - eps) 

    loss = - ( y_true*np.log(p) + (1-y_true)*np.log(1-p) )
    

    return loss.tolist()