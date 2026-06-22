def max_pool_2d(image, kernel_size, stride):
    """
    Returns: 2D list of shape (H_out, W_out), max-pooled values rounded to 4 decimals
    """
    import numpy as np 
    from numpy.lib.stride_tricks import sliding_window_view as sld

    image = np.asarray(image, dtype=np.float64)

    H, W = image.shape

    H_out = (H-kernel_size)//stride + 1 
    W_out = (W-kernel_size)//stride + 1

    output = np.zeros((H_out, W_out), dtype=np.float64)

    windows = sld(image, window_shape=(kernel_size, kernel_size))

    patches = windows[::stride, ::stride, :, :]
    
    output = np.max(patches, axis=(2,3))

    return np.round(output,4).tolist()

    
