import numpy as np

def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """
    # Write code here
    X = np.asarray(X, float)
    N, D = X.shape

    X_c = X - np.mean(X, axis=0)

    cov = X_c.T @ X_c / (N-1)

    eig_vals, eig_vectors = np.linalg.eigh(cov)

    index = np.argsort(eig_vals)[::-1][:k]

    W = eig_vectors[:,index]

    proj = np.dot(X_c, W)
  
    return proj