import numpy as np

def local_response_normalization(x: np.ndarray, k: float = 2, n: int = 5,
                                  alpha: float = 1e-4, beta: float = 0.75) -> np.ndarray:
    """
    Apply Local Response Normalization across channels.
    """
    # YOUR CODE HERE
    B, H, W, C = x.shape

    def squared(x):
        return x**2

    b = np.zeros_like(x)

    for i in range(C):
        min_val = min(C, i+ n//2 +1)
        max_val = max(0, i-n//2)
    
        window = x[:,:,:,max_val:min_val]
        b[:,:,:,i] = x[:,:,:,i] / ( k + alpha * np.sum(squared(window), axis=3) )**beta

    return b

    