import numpy as np

def info_nce_loss(Z1: list, Z2: list, temperature: float = 0.1) -> float:
    """
    Returns the loss as a float.
    """
    # Write code here
    Z1 = np.asarray(Z1, dtype=np.float64)
    Z2 = np.asarray(Z2, dtype=np.float64)
    
    S = np.matmul(Z1, Z2.T) / temperature

    max_val = np.max(S)

    num = np.exp(np.diag(S - max_val))

    den = np.sum(np.exp(S - max_val), axis=1)

    l = -np.mean(np.log(num/den), axis=0)

    return float(l)

    