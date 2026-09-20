import numpy as np

def detect_skew(train_dist: dict, serving_dist: dict, threshold: float = 0.2, eps: float = 1e-10) -> dict:
    """
    Returns a dictionary of feature PSI scores and skew flags.
    """
    # Write code here
    out = {}
    
    for feature in train_dist:
        t = np.asarray(train_dist[feature], dtype=np.float64) + eps
        s = np.asarray(serving_dist[feature], dtype=np.float64) + eps
        psi = float(np.sum((s-t)*np.log(s/t)))
        skewed = bool(psi >= threshold)
        d = {"psi":psi, "skewed":skewed}
        out[feature] = d


    return out