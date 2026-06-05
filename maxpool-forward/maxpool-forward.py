def maxpool_forward(X, pool_size, stride):
    """
    Compute the forward pass of 2D max pooling.
    """
    # Write code here
    import numpy as np 


    X = np.array(X, dtype=np.float64)

    H, W = X.shape

    H_out = (H-pool_size)//stride + 1
    W_out = (W-pool_size)//stride + 1

    out = np.zeros((H_out, W_out), dtype=np.float64)

    for i in range(H_out):
        for j in range(W_out):
            window = X[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
            out[i,j] = np.max(window)


    return out.tolist()