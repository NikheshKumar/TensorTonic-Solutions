import numpy as np

def unet_output(features: np.ndarray, num_classes: int) -> np.ndarray:
    """
    U-Net output layer: 1x1 conv for pixel-wise classification.
    Preserves spatial dims, changes channels to num_classes.
    Returns zero array with correct shape.
    """
    # Your implementation here

    features = np.asarray(features, np.float64)

    B, H, W, C_feat = features.shape

    out_shape = B, H, W, num_classes

    output = np.zeros((out_shape), dtype=np.float64)

    return output
