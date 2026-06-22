def pad_and_center_crop(image, pad, crop_h, crop_w):
    """
    Returns: 2D list of lists of floats with shape (crop_h, crop_w), each rounded to 4 decimals
    """
    import numpy as np 

    image = np.asarray(image, dtype=np.float64)
    H, W = image.shape

    canvas = np.pad(image, pad_width=pad, mode='constant', constant_values=0)

    r_start = (H + 2*pad - crop_h) // 2
    c_start = (W + 2*pad - crop_w) // 2

    ans = canvas[r_start:r_start+crop_h, c_start:c_start+crop_w]

    return np.round(ans, 4).tolist()

    

    

    

    
