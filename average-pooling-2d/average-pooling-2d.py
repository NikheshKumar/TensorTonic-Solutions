def average_pooling_2d(X, pool_size):
    """
    Apply 2D average pooling with non-overlapping windows.
    """
    # Write code here
    import numpy as np 

    X = np.asarray(X, float)

    H, W = X.shape

    H_out = H // pool_size
    W_out = W // pool_size
    p = pool_size

    out = np.zeros((H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            out[i][j] = (1/p**2) * np.sum(X[i*p : i*p + p, j*p : j*p + p])

    return out.tolist()        

    
