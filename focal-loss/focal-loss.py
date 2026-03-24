import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.
    """
    # Write code here
    p = np.asarray(p, float)
    y = np.asarray(y, float)

    eps = 1e-8
    p = np.clip(p, eps, 1-eps)

    fl = -((1-p)**gamma)*y*np.log(p) - (p**gamma)*(1-y)*np.log(1-p)

    return float(np.mean(fl))