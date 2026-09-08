import numpy as np

def kl_divergence(p: list, q: list, eps: float = 1e-12) -> float:
    """
    Returns the divergence as a float.
    """
    # Write code here
    p = np.array(p, dtype=np.float64)
    q = np.array(q, dtype=np.float64)

    mask = p>eps

    q = np.clip(q, eps, 1)

    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))
    