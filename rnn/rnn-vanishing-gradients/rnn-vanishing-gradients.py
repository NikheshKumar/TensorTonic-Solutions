import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    """
    Simulate gradient norm decay over T time steps.
    Returns list of gradient norms.
    """
    # YOUR CODE HERE
    n = 1.0
    norms = []
    sp_norm = np.linalg.norm(W_hh, ord=2)
  
    for t in range(T):
      norms.append(n)
      n = n * sp_norm
      
    return norms