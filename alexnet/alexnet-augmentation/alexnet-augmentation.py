import numpy as np

def random_crop(image: np.ndarray, crop_size: int = 224, crop_y: int = None, crop_x: int = None) -> np.ndarray:
    """
    Extract a crop from the image at (crop_y, crop_x). If not given, choose randomly.
    """
    # YOUR CODE HERE
    H, W, C = image.shape
    
    if crop_x is None:
        max_x = W - crop_size
        crop_x = np.random.randint(0, max_x + 1)
        
    if crop_y is None:
        max_y = H - crop_size
        crop_y = np.random.randint(0, max_y + 1)

    out = image[crop_y: crop_y + crop_size, crop_x: crop_x + crop_size, :]

    return out
    

def random_horizontal_flip(image: np.ndarray, p: float = 0.5, flip_rand: float = None) -> np.ndarray:
    """
    Flip image horizontally if flip_rand < p. If flip_rand not given, generate randomly.
    """
    # YOUR CODE HERE
    if flip_rand is None:
        flip_rand = np.random.default_rng().random()

    out = image

    if flip_rand < p:
        out = np.fliplr(image)

    return out