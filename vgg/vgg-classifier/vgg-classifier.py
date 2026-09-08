import numpy as np

def vgg_classifier(features: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                   W2: np.ndarray, b2: np.ndarray,
                   W3: np.ndarray, b3: np.ndarray) -> np.ndarray:
    """
    Returns float64 class logits with shape (B, C_classes).
    """
    #flatten
    N, H, W, C = features.shape
    features = features.reshape(N, -1)

    #affine1
    out = features @ W1 + b1
    #relu1
    out = np.maximum(0, out)

    #affine2
    out = out @ W2 + b2
    #relu2
    out = np.maximum(0, out)

    #affine3
    out = out @ W3 + b3

    return out
    