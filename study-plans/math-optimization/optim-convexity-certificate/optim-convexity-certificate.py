import numpy as np

def convexity_certificate(H):
    """
    Returns: dict with 'is_convex' (bool) and 'min_eigenvalue' (float, rounded to 6 decimals)
    """

    H = np.array(H, dtype=np.float64)

    eig_vals = np.linalg.eigvals(H)

    min_evalue = np.min(eig_vals)

    is_convex = bool(min_evalue >= -1e-6)
    
    return {'is_convex':is_convex, 'min_eigenvalue':min_evalue}
