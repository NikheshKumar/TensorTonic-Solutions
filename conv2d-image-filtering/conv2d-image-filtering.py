def conv2d(image, kernel, stride=1, padding=0):
    """
    Apply 2D convolution to a single-channel image.
    """
    # Write code here
    import numpy as np 

    image = np.asarray(image, np.float64)
    kernel = np.asarray(kernel, np.float64)

    H, W = image.shape 
    kh, kw = kernel.shape
    

    if padding>0:
        padded = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')
    else:
        padded = image

    H_p, W_p = padded.shape
    H_out, W_out = 1 + (H_p - kh) // stride, 1 + (W_p - kw) // stride 

    output = np.zeros((H_out, W_out), np.float64)
    

    for i in range(H_out):
        for j in range(W_out):
            padded_patch = padded[i*stride :i*stride + kh, j*stride: j*stride + kw]
            output[i,j] = np.sum(padded_patch * kernel)


    return output.tolist()

    