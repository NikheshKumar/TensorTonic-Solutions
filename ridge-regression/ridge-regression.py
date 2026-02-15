def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.
    """
    # Write code here
    import numpy as np
    X = np.asarray(X)
    y = np.asarray(y)

    Z = lam*np.eye(len(X[1]), len(X[0]))
    w = (np.linalg.inv(X.T @X + Z) )@X.T @y

    return w.tolist()