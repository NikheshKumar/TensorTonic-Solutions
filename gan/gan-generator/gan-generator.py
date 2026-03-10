import numpy as np

def generator(z: np.ndarray, output_dim: int) -> np.ndarray:
    """
    Generate fake data from noise vectors.
    """
    # Your implementation here
  
    z = np.asarray(z, float)
    eps = 1e-3

    def leaky_relu(x, alpha=0.2):
      return np.maximum(alpha * x, x)
  
    # linear + leaky relu 
    W1 = np.random.uniform(-eps, eps, size=(z.shape[1], z.shape[1]))
    b1 = np.random.uniform(-eps, eps, size=(z.shape[1],))
    h1 = leaky_relu(z @ W1 + b1)

    # linear + leaky relu
    W2 = np.random.uniform(-eps, eps, size=(h1.shape[1], h1.shape[1]))
    b2 = np.random.uniform(-eps, eps, size=(h1.shape[1],))
    h2 = leaky_relu(h1 @ W2 + b2)

    # linear + tanh
    W3 = np.random.uniform(-eps, eps, size=(h2.shape[1], output_dim))
    b3 = np.random.uniform(-eps, eps, size=(output_dim,))
    x_hat = np.tanh(h2 @ W3 + b3)

    # fake image

    return x_hat

    
    