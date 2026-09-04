import numpy as np

def generator(z: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Returns generated samples as a float64 array with shape (B, D).
    """
    return np.tanh(z@W +b)