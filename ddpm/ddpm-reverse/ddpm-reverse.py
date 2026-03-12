import numpy as np

def reverse_step(
    x_t: np.ndarray,
    t: int,
    epsilon_pred: np.ndarray,
    betas: np.ndarray
) -> np.ndarray:
    """
    Perform one reverse diffusion step.
    """
    # YOUR CODE HERE
    x_t = np.asarray(x_t, float)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)

    z = np.random.randn(*x_t.shape)

    mu = (x_t - (1-alphas[t])*epsilon_pred/(np.sqrt(1-alpha_bars[t])) ) / np.sqrt(alphas[t])

    var = betas[t]

    return mu + np.sqrt(var)*z if t>1 else mu
  