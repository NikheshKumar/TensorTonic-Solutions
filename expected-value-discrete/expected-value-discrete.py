import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    x, p = np.asarray(x, int), np.asarray(p, float)

    E = np.sum(x*p)

    tot = np.sum(p)
    if abs(tot - 1.0) > 1e-6:
        raise ValueError("Probabilities do not sum to 1 (within tolerance)")

    return E    
