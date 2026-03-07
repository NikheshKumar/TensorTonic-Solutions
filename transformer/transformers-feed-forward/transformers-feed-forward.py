import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Apply position-wise feed-forward network.
    """
    # Your code here
  
    x = np.asarray(x, float)
    W1 = np.asarray(W1, float)
    W2 = np.asarray(W2, float)

    relu = np.maximum(0,np.dot(x,W1)+b1)

    ffn_output = relu @ W2 + b2

    return ffn_output