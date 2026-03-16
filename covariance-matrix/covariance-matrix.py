import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Write code here
    X = np.asarray(X, float)

    if X.ndim !=2:
        return None 

    if X.size == 0:
        return None

    N, D = X.shape

    if N < 2:
        return None

    if D == 0:
        return np.zeros((0, 0)) 
      
    mu = np.mean(X, axis=0)

    X_c = X - mu

    cov = (X_c.T @ X_c) / (N-1)

    return cov