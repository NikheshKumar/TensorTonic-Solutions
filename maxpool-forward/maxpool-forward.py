def maxpool_forward(X, pool_size, stride):
    """
    Compute the forward pass of 2D max pooling.
    """
    # Write code here
    import numpy as np 

    X = np.asarray(X)
    H, W = X.shape

    h_out = int( np.ceil( (H-pool_size )/stride) ) + 1
    w_out = int( np.ceil( (W-pool_size )/stride) ) + 1

    output = np.zeros((h_out, w_out), float)

    for i in range(h_out):
        for j in range(w_out):
            output[i,j] = np.max(X[i*stride : i*stride + pool_size, j*stride : j*stride + pool_size])

    return output.tolist()        




