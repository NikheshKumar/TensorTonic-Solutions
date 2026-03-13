import numpy as np

def impute_missing(X, strategy='mean'):
    """
    Fill NaN values in each feature column using column mean or median.
    """
    # Write code here
    X = np.atleast_2d(np.asarray(X, float))
    N, D = X.shape

    X = X.reshape(-1,1) if N==1 else X
  
    if strategy=='mean':
      mask = np.isnan(X)
      mean_vals = np.nanmean(X, axis=0)
      mean_vals = np.where(np.isnan(mean_vals), 0.0, mean_vals)
      X_new = np.where(mask, mean_vals, X)
      
    if strategy=='median':
      mask = np.isnan(X)
      median_vals = np.nanmedian(X, axis=0)
      median_vals = np.where(np.isnan(median_vals), 0.0, median_vals)
      X_new = np.where(mask, median_vals, X)

    X_new = X_new.flatten() if N==1 else X_new

    return X_new
      