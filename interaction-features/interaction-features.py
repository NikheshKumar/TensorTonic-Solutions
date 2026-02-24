def interaction_features(X):
    """
    Generate pairwise interaction features and append them to the original features.
    """
    # Write code here
    import numpy as np

    X = np.asarray(X)

    a,b = np.indices((X.shape[1], X.shape[1]))

    mask = a<b

    rows = a[mask]
    cols = b[mask]
    ans = []

    for i,j in zip(rows, cols):
      p = X[:,i] * X[:,j]
      ans.append(p)

    if ans:
      res = np.column_stack([X] + ans)
      return res.tolist()

    else:
      return X.tolist()
    
  