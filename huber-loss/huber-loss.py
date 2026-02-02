import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """
    # Write code here
    y_pred, y_true = np.asarray(y_pred, dtype=float), np.asarray(y_true, dtype=float)
    e = np.abs(y_pred-y_true)

    loss = np.where(
        e <= delta,
        0.5 * (e**2),
        delta * (e - (0.5 * delta))
    )   

    return np.mean(loss)
