def normalize_image(image, mean, std):
    """
    Returns: 3D list of shape (H, W, C), each value rounded to 4 decimals
    """
    import numpy as np

    image = np.asarray(image, dtype=np.float64)


    image_normalized = (image - mean) / std

    return image_normalized
