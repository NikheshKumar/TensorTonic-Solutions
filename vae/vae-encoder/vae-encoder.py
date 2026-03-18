import numpy as np

def vae_encoder(x: np.ndarray, latent_dim: int) -> tuple:
    """
    Encode input to latent distribution parameters.
    """
    # Your implementation here
    x = np.asarray(x, float)

    eps = 0.05
    W1 = np.random.randn(x.shape[1], latent_dim) * eps
    b1 = np.random.randn(latent_dim)*eps

    h = np.maximum(0, x@W1 + b1)

    W_mu = np.random.randn(latent_dim, latent_dim) * eps
    b_mu = np.random.randn(latent_dim)*eps
    mu = h@W_mu + b_mu

    W_log_var = np.random.randn(latent_dim, latent_dim) * eps
    b_log_var = np.random.randn(latent_dim)*eps
    log_var = h@W_log_var + b_log_var

    return mu, log_var