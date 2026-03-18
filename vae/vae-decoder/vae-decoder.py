import numpy as np

def vae_decoder(z: np.ndarray, output_dim: int) -> np.ndarray:
    """
    Decode latent vectors to reconstructed data.
    """
    # Your implementation here
    eps = 0.05
  
    W1 = np.random.randn(z.shape[1], output_dim) * eps
    b1 = np.random.randn(output_dim) * eps
    h = np.maximum(0, z@W1 + b1)

    def _sigmoid(x):
      return np.exp(x) / (1+np.exp(x))

    W2 = np.random.randn(output_dim, output_dim) * eps
    b2 = np.random.randn(output_dim) * eps
    x_hat = _sigmoid(h@W2 + b2)

    return x_hat