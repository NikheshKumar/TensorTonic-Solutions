def morphological_op(image, kernel, operation):
    """
    Apply morphological erosion or dilation to a binary image.
    """
    # Write code here
    import numpy as np 

    image = np.asarray(image, np.float64)
    kernel= np.asarray(kernel, np.float64)

    k_h, k_w = kernel.shape
    h, w = image.shape

    pad_h, pad_w = k_h//2, k_w//2

    pad = ((pad_h, pad_h), (pad_w, pad_w))

    padded_image = np.pad(image, pad, mode='constant', constant_values=0)

    res = np.zeros_like(image, np.float64)

    window = np.lib.stride_tricks.sliding_window_view(padded_image, (k_h, k_w))

    if operation=="dilate":

        masked_window = np.where(kernel==1, window, -np.inf)

        res = np.max(masked_window, axis=(2,3))

        
    if operation=="erode":

        masked_window = np.where(kernel==1, window, np.inf)

        res = np.min(masked_window, axis=(2,3))

    return res.tolist()

        

        

        

        

        

    