import numpy as np

def alexnet_conv1(image: np.ndarray) -> np.ndarray:
    """
    AlexNet first conv layer: 11x11, stride 4, 96 filters (shape simulation).
    """
    # YOUR CODE HERE
    B, H_in, W_in, C = image.shape
    s = 4
    p = 2
    k = 11
    F = 96
    H_out = (H_in + 2*p - k) // s + 1
    W_out = (W_in + 2*p - k) //s + 1

    out = np.zeros((B,H_out,W_out,F))
    return out

    
    

    