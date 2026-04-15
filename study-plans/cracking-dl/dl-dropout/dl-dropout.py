import numpy as np

def dropout(X, mask, drop_prob, mode):
    """
    Returns: 2D list with values rounded to 4 decimal places.
    """
    X = np.asarray(X, np.float64)
    mask = np.asarray(mask, dtype=np.float64)

    out = X

    if mode=="test" or drop_prob==0.0:
        out = X
        
    if mode=="train" and drop_prob > 0.0:
        out = X * mask / (1-drop_prob)


    return np.round(out, 4).tolist()