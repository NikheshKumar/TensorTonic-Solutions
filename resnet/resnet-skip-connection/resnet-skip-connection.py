import numpy as np

def compute_gradient_with_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITH skip connections.
    Gradient at layer l = sum of paths through network
    """
    # YOUR CODE HERE
    gradients_F = np.asarray(gradients_F, float)
    x_new = np.asarray(x, float)
    
    for grad in reversed(gradients_F):
      id = np.eye(grad.shape[0])
      x_new = x_new @ (grad + id)
      
    return x_new
    

def compute_gradient_without_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITHOUT skip connections.
    """
    # YOUR CODE HERE
    gradients_F = np.asarray(gradients_F, float)
    x_new = np.asarray(x, float)
  
    for grad in reversed(gradients_F):
      x_new = x_new @ grad

    return x_new
      
