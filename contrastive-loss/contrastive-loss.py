import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    """
    a, b: arrays of shape (N, D) or (D,)  (will broadcast to (N,D))
    y:    array of shape (N,) with values in {0,1}; 1=similar, 0=dissimilar
    margin: float > 0
    reduction: "mean" (default) or "sum"
    Return: float
    """
    # Write code here
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    y = np.asarray(y, float)

    if a.ndim == 1:
        a = a[np.newaxis, :]
    if b.ndim == 1:
        b = b[np.newaxis, :]

    d = np.linalg.norm(a-b, axis=1)

    if reduction=="mean":
        loss = np.mean( y * d**2 + (1-y) * np.maximum(0, margin-d)**2 )

    elif reduction=="sum":  
        loss = np.sum( y * d**2 + (1-y) * np.maximum(0, margin-d)**2 )

    else:
        raise ValueError("reduction must be either 'mean' or 'sum'")    

    return float(loss) 

