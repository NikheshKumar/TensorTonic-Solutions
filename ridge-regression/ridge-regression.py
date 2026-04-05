def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.
    """
    # Write code here
    import numpy as np 

    X = np.asarray(X, float)
    y = np.asarray(y, float)

    m,n = X.shape

    id = np.eye(n,n)

    beta = np.linalg.inv(X.T@X + lam*id)@X.T@y
    
    return beta.tolist()


    