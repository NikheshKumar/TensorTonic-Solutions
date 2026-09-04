import numpy as np

def train_discriminator_step(
    real_data: np.ndarray,
    fake_data: np.ndarray,
    D_W: np.ndarray,
    learning_rate: float,
) -> dict:
    """
    Returns updated discriminator weights and the pre-update loss.
    """
    
    def sigmoid(z):
        return np.exp(z)/ (1.0+np.exp(z))

    eps = 1e-8
        
    p_real = np.clip(sigmoid(real_data@D_W), eps, 1-eps)
    p_fake = np.clip(sigmoid(fake_data@D_W), eps, 1-eps)

    loss_D = -np.mean( np.log(p_real) + np.log(1.0-p_fake) )

    N = p_real.shape[0]

    grad = (real_data.T@(p_real - 1.0) + (fake_data).T@p_fake) / N

    W_new = D_W - learning_rate * grad

    return {"new_discriminator_weights":W_new,"discriminator_loss":loss_D}