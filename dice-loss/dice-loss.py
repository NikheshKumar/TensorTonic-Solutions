import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    """
    # Write code here
    p = np.asarray(p)
    y = np.asarray(y)

    p = p.flatten()
    y = y.flatten()

    num = 2 * np.sum(p*y) + eps

    den = np.sum(p) + np.sum(y) + eps

    return 1 - (num / den)