import numpy as np

def compute_ddpm_loss(
    model_predict: callable,
    x_0: np.ndarray,
    betas: np.ndarray,
    T: int
) -> float:
    """
    Compute DDPM training loss for a batch of images.
    """
    # YOUR CODE HERE
    x_0 = np.asarray(x_0, float)
    batch_size = x_0.shape[0]
  
    alphas = 1-betas
    alpha_bars = np.cumprod(alphas)

    t = np.random.randint(1,T,(batch_size,))
    alpha_bar_t = alpha_bars[t]

    new_shape = (-1,) + (1,)*(x_0.ndim - 1)
    sqrt_alpha_bars = np.sqrt(alpha_bar_t.reshape(new_shape))
    sqrt_one_minus_alpha_bars = np.sqrt(1 - alpha_bar_t.reshape(new_shape))

    eps = np.random.standard_normal(size=x_0.shape)

    xt = sqrt_alpha_bars * x_0 + sqrt_one_minus_alpha_bars * eps
    eps_pred = model_predict(xt, t)

    mse = np.mean((eps-eps_pred)**2)

    return mse
