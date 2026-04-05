import numpy as np

def _entropy(y):
    """
    Helper: Compute Shannon entropy (base 2) for labels y.
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    vals, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0

def information_gain(y, split_mask):
    """
    Compute Information Gain of a binary split on labels y.
    Use the _entropy() helper above.
    """
    # Write code here
    y = np.asarray(y, dtype=float)
    split_mask = np.asarray(split_mask, dtype=bool)

    cla, counts = np.unique(y, return_counts=True)

    H = _entropy(y)

    y_L = y[split_mask]
    H_L = _entropy(y_L)

    y_R = y[~split_mask]
    H_R = _entropy(y_R)

    N = len(y)
    n_L = len(y_L)
    n_R = N - n_L
    

    ig = H - (n_L * H_L + n_R * H_R) / N

    return ig
    
