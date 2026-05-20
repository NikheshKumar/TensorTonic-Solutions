import numpy as np

def pooling(input, pool_size, stride, pool_type):
    """
    Returns: 3D list with pooled values rounded to 4 decimal places.
    """

    input = np.asarray(input, np.float64)

    C, H, W = input.shape

    H_out = (H-pool_size)//(stride) + 1
    W_out = (W-pool_size)//(stride) + 1

    C_stride, H_stride, W_stride = input.strides

    new_strides = (C_stride, H_stride * stride, W_stride * stride, H_stride, W_stride )

    output = np.zeros((C, H_out, W_out), np.float64)


    windows = np.lib.stride_tricks.as_strided(input, shape=(C, H_out, W_out, pool_size, pool_size), strides=new_strides, writeable=False)
    
    if pool_type == "max":
        output = np.max(windows, axis=(-2,-1))
        
    elif pool_type == "average":
        output = np.mean(windows, axis=(-2,-1))


    return [[[round(float(output[c,i,j]), 4) for j in range(W_out)] for i in range(H_out)] for c in range(C)]
        