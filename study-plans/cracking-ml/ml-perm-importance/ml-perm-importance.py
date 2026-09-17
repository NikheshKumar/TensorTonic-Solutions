import numpy as np

def permutation_importance(X, y, predict_fn, n_repeats=5, seed=42):
    """
    Returns: list of importance scores (one per feature) rounded to 4 decimal places
    """

    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    n, d = X.shape

    rng = np.random.RandomState(seed)

    def score(X, y):
        return np.mean(predict_fn(X)==y)

    baseline = score(X, y)

    imp = np.zeros((d,), dtype=np.float64)
    
    for i in range(d):
        s = []
        for j in range(n_repeats):
            X_new = X.copy()
            X_new[:, i] = rng.permutation(X_new[:, i])
            s.append(baseline - score(X_new, y))
            imp[i] = np.mean(s)

    return [round(float(i),4) for i in imp]
            

    

    
    
