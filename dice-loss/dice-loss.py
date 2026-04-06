import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    """
    # Write code here
    p = np.asarray(p, float).ravel()
    y = np.asarray(y, float).ravel()


    num = 2*np.sum(p*y) + eps

    den = np.sum(p) + np.sum(y) + eps

    loss = 1 - num/den

    return loss