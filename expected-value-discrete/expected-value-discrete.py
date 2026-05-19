import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    x = np.asarray(x, np.float64)
    p = np.asarray(p, np.float64)

    if x.shape != p.shape:
        raise ValueError("x and p should have same shape")

    if not np.allclose(np.sum(p), 1.0, atol=1e-6):
        raise ValueError("Probabilities must sum to 1")

    return float(np.sum(x*p))