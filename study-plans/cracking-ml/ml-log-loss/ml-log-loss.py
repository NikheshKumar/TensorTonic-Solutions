import numpy as np

def log_loss(y_true, y_pred):
    """
    Returns: float
    """
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1-eps)
    y_true = np.clip(y_true, eps, 1-eps)

    l = -np.mean(y_true * np.log(y_pred) + (1-y_true)*np.log(1-y_pred))

    return round(float(l),4)

    
