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
    a = np.array(a, float, ndmin=2)
    b = np.array(b, float, ndmin=2)
    y = np.array(y, float, ndmin=2)

    d = np.linalg.norm(a-b, axis=1)

    l = np.sum(y*(d**2) + (1-y)*(np.maximum(0,margin-d)**2)) if reduction=="sum" else np.mean(y*(d**2) + (1-y)*(np.maximum(0,margin-d)**2))

    return float(l)
    