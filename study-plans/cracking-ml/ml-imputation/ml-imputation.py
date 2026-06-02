import numpy as np

def impute(X, method="mean"):
    """
    Returns: 2D list with NaN values replaced using the specified method
    """
    X = np.array(X, dtype=np.float64)

    m,n = X.shape

    mask = np.isnan(X)

    if method == "mean":
        m = np.nanmean(X, axis=0)
        
    if method == "median":
        m = np.nanmedian(X, axis=0)

    if method == "mode":
        
        m = np.zeros((n,), dtype=np.float64)
        for i in range(n):
            cols = X[:, i]
            valid_cols = cols[~np.isnan(cols)]
            vals, counts = np.unique(valid_cols, return_counts=True)
            m[i] = vals[np.argmax(counts)]

    
    m = np.nan_to_num(m, nan=0.0)

    for i in range(n):
        X[mask[:, i], i] = m[i]
        

    return X