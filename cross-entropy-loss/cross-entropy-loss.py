import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    eps = 1e-7
    y_pred = np.clip(y_pred, eps, 1 - eps)

    conf = y_pred[np.arange(y_pred.shape[0]), y_true]

    loss = np.log(conf)

    ce = -np.mean(loss)

    return ce