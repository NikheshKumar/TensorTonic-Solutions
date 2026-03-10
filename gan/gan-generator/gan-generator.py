import numpy as np

def generator(z: np.ndarray, output_dim: int) -> np.ndarray:
    """
    Generate fake data from noise vectors.
    """
    # Your implementation here
  
    z = np.asarray(z, float)
    eps = 1e-3

    def leaky_relu(x, alpha=1e-3):
      return np.maximum(alpha * x, x)
  
    # linear + leaky relu 
    W1 = np.random.uniform(-eps, eps, (z.shape[1], output_dim))
    b1 = np.random.uniform(-eps, eps, (output_dim))
    h1 = leaky_relu(z @ W1 + b1, alpha=1e-3)

    # linear + leaky relu
    W2 = np.random.uniform(-eps, eps, (h1.shape[1], output_dim))
    b2 = np.random.uniform(-eps, eps, (output_dim))
    h2 = leaky_relu(h1 @ W2 + b2, alpha=1e-3)

    # linear + tanh
    W3 = np.random.uniform(-eps, eps, (h2.shape[1], output_dim))
    b3 = np.random.uniform(-eps, eps, (output_dim))
    x_hat = np.tanh(h2 @ W3 + b3)

    # fake image

    return x_hat

    
    