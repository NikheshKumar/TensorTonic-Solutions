import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    # Write code here
    
    y_pred = np.asarray(y_pred, float)
    y_true = np.asarray(y_true, float)

    n = len(y_pred)
    
    if y_pred.shape != y_true.shape:
        return None

    se = np.sum( (y_pred-y_true)**2 )

    mse = se / n

    return float(mse)
