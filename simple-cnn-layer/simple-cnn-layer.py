import numpy as np
from numpy.lib.stride_tricks import as_strided

def conv2d(x, W, b):
    """
    Simple 2D convolution layer forward pass.
    Valid padding, stride=1.
    """
    # Write code here
    x = np.asarray(x, float)
    W = np.asarray(W, float)
    b = np.asarray(b, float)

    N, C_in, H_in, W_in = x.shape
    C_out, C_in, KH, KW = W.shape

    H_out = H_in - KH + 1 
    W_out = W_in - KW + 1

    y = np.zeros((N, C_out, H_out, W_out), float)

    stride = 1
    
    for n in range(N):
        
        x_img = x[n]
        cimg, himg, wimg = x_img.strides
        patch_stride = (himg, wimg, cimg, himg, wimg)
        patch_shape = (H_out, W_out, C_in, KH, KW)
        x_patch = as_strided(x_img, shape=patch_shape, strides=patch_stride)
        
        for cout in range(C_out):
        
            W_patch = W[cout]
            y[n, cout] = np.sum(x_patch*W_patch, axis=(2, 3, 4)) + b[cout]

    return y
            
        
        
    