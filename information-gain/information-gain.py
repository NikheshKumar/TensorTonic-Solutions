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
    y = np.asarray(y, np.float64)
    split_mask = np.asarray(split_mask, bool)

    H = _entropy(y)

    Yl = y[split_mask]
    Hl = _entropy(Yl)

    Yr = y[~split_mask]
    Hr = _entropy(Yr)

    N = len(y)
    n_l = len(Yl)
    n_r =len(Yr)

    ig = H - (n_l*Hl + n_r*Hr) / N

    return ig

    

    
    
