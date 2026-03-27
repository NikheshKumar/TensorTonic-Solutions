import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    """
    # Write code here
    p = np.array(p, float).ravel()
    y = np.array(y, float).ravel()

    num = 2*np.sum(p*y) + eps 
    den = np.sum(p) + np.sum(y) + eps

    score = num / den

    loss = 1 - score

    return float(loss)