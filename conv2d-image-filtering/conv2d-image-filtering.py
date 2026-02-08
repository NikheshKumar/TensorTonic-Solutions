def conv2d(image, kernel, stride=1, padding=0):
    """
    Apply 2D convolution to a single-channel image.
    """
    # Write code here
    import numpy as np 

    image = np.asarray(image, float)
    kernel = np.asarray(kernel, float)

    H,W = image.shape
    kh, kw = kernel.shape

    if padding > 0:
        image_padded = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')
    else:
        image_padded = image

    H_p, W_p = image_padded.shape

    H_out = 1 + (H_p - kh) // stride
    W_out = 1 + (W_p - kw) // stride 

    out = np.zeros((H_out, W_out), float)   

    for i in range(H_out):
        for j in range(W_out):
            region = image_padded[i*stride :i*stride + kh, j*stride: j*stride + kw]
            out[i][j] = np.sum(region*kernel)


    return out.tolist()                