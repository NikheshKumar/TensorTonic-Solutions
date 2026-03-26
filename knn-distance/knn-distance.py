import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise distances and return k nearest neighbor indices.
    """
    # Write code here
    X_train = np.asarray(X_train, float)
    X_test = np.asarray(X_test, float)

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1,1)
    if X_test.ndim==1:
        X_test = X_test.reshape(-1,1)


    X_train_sq = np.sum(X_train**2, axis=1)
    X_test_sq = np.sum(X_test**2, axis=1, keepdims=True)
    dot_prod = 2 * (X_test @ X_train.T)

    pairwise_dist = np.maximum(X_test_sq + X_train_sq - dot_prod, 0)

    indices = np.argsort(pairwise_dist, axis=1)[:, :k]

    if k <= X_train.shape[0]:
        return indices[:, :k]
        
    res = np.full((X_test.shape[0], k), -1, dtype=int)
    res[:, :X_train.shape[0]] = indices
    
    return res
    