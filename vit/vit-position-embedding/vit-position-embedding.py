import numpy as np

def add_position_embedding(patches: np.ndarray, num_patches: int, embed_dim: int, pos_embed: np.ndarray) -> np.ndarray:
    """
    Returns the float64 tokens after adding position embeddings.
    """
    z = patches + pos_embed

    return z