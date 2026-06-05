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

    s_H, s_W = X.strides

    tensor_shape = (H_out, W_out, pool_size, pool_size)
    strides_shape = (stride * s_H, stride * s_W, s_H, s_W)

    windows = np.lib.stride_tricks.as_strided(
        X, shape=tensor_shape, strides=strides_shape, writeable=False
    )

    out = np.max(windows, axis=(-2,-1))


    return out.tolist()