import numpy as np

def prepend_class_token(patches: np.ndarray, embed_dim: int, cls_token: np.ndarray) -> np.ndarray:
    """
    Returns the float64 sequence with the class token at position zero.
    """
    B, N, D = patches.shape
    
    cls_token = np.broadcast_to(cls_token,(B,1,D))

    z = np.concatenate([cls_token, patches], axis=1)

    return z