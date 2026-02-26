import numpy as np

def pearson_correlation(X):
    """
    Compute Pearson correlation matrix from dataset X.
    """
    # Write code here

    X = np.asarray(X, float)
  
    if X.ndim!=2:
      return None

    N,d = X.shape
    if N<2:
      return None


    X_new = X - np.mean(X, axis=0, keepdims=True)
  
    cov = np.dot(X_new.T, X_new) / (N-1)

    sdev = np.std(X, axis=0, ddof=1, keepdims=True)

    den =  np.outer(sdev,sdev)

    corr = np.divide(cov, den, out=np.full((d, d), np.nan), where=(den!=0) ) 

    mask = np.diag(den)!=0
    indices = np.where(mask)[0]
    for i in indices:
        corr[i, i] = 1.0
 
    return corr.tolist()