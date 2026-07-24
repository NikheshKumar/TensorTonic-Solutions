import numpy as np

def classify_critical_point(H):
    """
    Returns: one of 'local_min', 'local_max', 'saddle', 'degenerate'
    """
    H = np.asarray(H, dtype=np.float64)
    
    evals = np.linalg.eigvals(H)

    if np.min(evals) > 1e-6:
        return "local_min"

    elif np.max(evals) < -1e-6:
        return "local_max"

    elif np.any(np.abs(evals) <= 1e-6):
        return "degenerate"

    elif np.max(evals) >  1e-6 and np.min(evals) < 1e-6 and np.min(np.abs(evals)):
        return "saddle"

    else:
        return "degenerate"
    
        
