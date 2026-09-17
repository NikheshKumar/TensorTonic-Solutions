import numpy as np

def pca(X, n_components=2):
    """
    Returns: tuple of (transformed_data, explained_variance_ratios)
    """
    X = np.array(X, dtype=np.float64)
    n, d = X.shape

    mu = X.mean(axis=0)
    X_new = X - mu

    evals, evecs = np.linalg.eigh((1.0/(n-1)) * X_new.transpose() @ X_new)

    order = np.argsort(evals)[::-1]
    evals_new = evals[order]
    evecs_new = evecs[:, order]

    W = evecs_new[:, :n_components].copy()

    for j in range(W.shape[1]):
        col = W[:, j]
        if col[np.argmax(np.abs(col))] < 0:
            W[:, j] = -col
    
    Z = X_new @ W

    var_ratios = evals_new[:n_components] / np.sum(evals_new)


    return np.round(Z, 4).tolist(), np.round(var_ratios, 4).tolist()