import numpy as np

def pooling(input, pool_size, stride, pool_type):
    """
    Returns: 3D list with pooled values rounded to 4 decimal places.
    """

    input = np.asarray(input, np.float64)

    C, H, W = input.shape

    H_out = (H-pool_size)//(stride) + 1
    W_out = (W-pool_size)//(stride) + 1

    output = np.zeros((C, H_out, W_out), np.float64)

    
    for c in range(C):
        for i in range(H_out):
            for j in range(W_out):
                if pool_type == "max":
                    m = n = pool_size
                    output[c,i,j] = np.max(input[c, i*stride:i*stride+m, j*stride:j*stride+n])
                elif pool_type == "average":
                    m = n = pool_size
                    output[c,i,j] = np.mean(input[c, i*stride:i*stride+m, j*stride:j*stride+n])


    return [[[round(float(output[c,i,j]), 4) for j in range(W_out)] for i in range(H_out)] for c in range(C)]
        