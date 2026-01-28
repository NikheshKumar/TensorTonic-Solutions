import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Write code here
    X = np.asarray(X)

    if X.ndim !=2:
        return None 

    N, D = X.shape
    
    if X.size == 0:
        return None

    if N < 2:
        return None

    if D == 0:
        return np.zeros((0, 0))    

    mu = np.mean(X,axis=0)
    X_cent = X - mu

    cov_mat = (X_cent.T @ X_cent) / (N - 1)
    return cov_mat
