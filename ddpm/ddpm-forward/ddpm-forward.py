import numpy as np

def get_alpha_bar(betas: np.ndarray) -> np.ndarray:
    """
    Compute cumulative product of (1 - beta).
    """
    # YOUR CODE HERE
    return np.cumprod(1.0-betas)

def forward_diffusion(
    x_0: np.ndarray,
    t: int,
    betas: np.ndarray
) -> tuple:
    """
    Sample x_t from q(x_t | x_0).
    """
    # YOUR CODE HERE
    x_0 = np.asarray(x_0, float)
    alpha_bar = get_alpha_bar(betas)
    eps = np.random.randn(*x_0.shape)

    x_t = x_0 * np.sqrt(alpha_bar[t]) + eps * np.sqrt(1-alpha_bar[t])

    return x_t, eps
