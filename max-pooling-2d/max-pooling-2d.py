def max_pooling_2d(X, pool_size):
    """
    Apply 2D max pooling with non-overlapping windows.
    """
    # Write code here
    import numpy as np 

    X = np.asarray(X, float)
    H,W = X.shape
    p = pool_size

    H_out = H // p
    W_out = W // p

    out = np.zeros((H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            out[i][j] = np.max(X[i*p : i*p + p, j*p : j*p + p])

    return out        