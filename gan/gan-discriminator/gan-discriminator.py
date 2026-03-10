import numpy as np

def discriminator(x: np.ndarray) -> np.ndarray:
    """
    Classify inputs as real or fake.
    """
    # Your implementation here
    x = np.asarray(x, float)
    eps = 1e-2

    def leaky_relu(x, alpha=0.2):
      return np.maximum(alpha*x, x)

    W1 = np.random.uniform(-eps, eps, (x.shape[1], x.shape[1]))
    b1 = np.random.uniform(-eps, eps, (x.shape[1],))
    h1 = leaky_relu(x @ W1 + b1)


    W2 = np.random.uniform(-eps, eps, (h1.shape[1], h1.shape[1]))
    b2 = np.random.uniform(-eps, eps, (h1.shape[1],))
    h2 = leaky_relu(h1 @ W2 + b2)


    W3 = np.random.uniform(-eps, eps, (h2.shape[1], 1))
    b3 = np.random.uniform(-eps, eps, (1,))
  
    y = h2 @ W3 + b3

    sig = 1 / (1 + np.exp(-np.clip(y, -500, 500)))

    return sig

    