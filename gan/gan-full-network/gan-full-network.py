import numpy as np

def gan_forward(
    z: np.ndarray,
    real_data: np.ndarray,
    G_W: np.ndarray,
    G_b: np.ndarray,
    D_W: np.ndarray,
) -> dict:
    """
    Returns generated samples, probabilities, and both GAN losses.
    """
    def sigmoid(z):
        return np.exp(z) / (1.0 + np.exp(z))
        
    x_hat = np.tanh(z@G_W + G_b)

    eps = 1e-8

    p_real = np.clip(sigmoid(real_data @ D_W), eps, 1-eps)

    p_fake = np.clip(sigmoid(x_hat @ D_W), eps, 1-eps)

    loss_D = -np.mean(np.log(p_real) + np.log(1.0-p_fake))

    loss_G = -np.mean(np.log(p_fake))

    return {"generated_samples":x_hat,"real_probabilities":p_real,"fake_probabilities":p_fake,"discriminator_loss":loss_D,"generator_loss":loss_G}

    