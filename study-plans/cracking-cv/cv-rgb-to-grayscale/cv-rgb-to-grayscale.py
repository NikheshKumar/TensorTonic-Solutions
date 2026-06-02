def rgb_to_grayscale(image):
    """
    Returns: 2D list of shape (H, W) with luma values rounded to 4 decimals
    """
    import numpy as np 

    image = np.array(image, dtype=np.float64)

    image_new = 0.299*image[:,:,0] + 0.587*image[:,:,1] + 0.114*image[:,:,2]

    return image_new
