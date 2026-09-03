import numpy as np

def patch_embed(image: np.ndarray, patch_size: int, embed_dim: int, W_proj: np.ndarray) -> np.ndarray:
    """
    Returns the float64 patch embeddings with shape (B, N, embed_dim).
    """
    B, H, W, C = image.shape

    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    if pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        B, H, W, C = image.shape

    N = H * W // patch_size**2

    image_patches = image.reshape(B, H//patch_size, patch_size, W//patch_size, patch_size, C)

    image_patches = image_patches.transpose(0, 1, 3, 2, 4, 5)

    image_patches = image_patches.reshape(B, N, patch_size * patch_size * C)

    bias = np.zeros((embed_dim,), dtype=np.float64)

    z = image_patches @ W_proj + bias 

    return z

    

    