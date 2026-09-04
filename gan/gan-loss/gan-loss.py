import numpy as np

def gan_losses(real_probs: np.ndarray, fake_probs: np.ndarray) -> dict:
    """
    Returns discriminator_loss and generator_loss as Python floats.
    """
    eps = 1e-8

    real_probs = np.clip(real_probs, eps, 1-eps)
    fake_probs = np.clip(fake_probs, eps, 1-eps)
    
    loss_d = -np.mean(np.log(real_probs) + np.log(1.0-fake_probs))

    loss_g =-np.mean(np.log(fake_probs))

    return {"discriminator_loss":float(loss_d), "generator_loss":float(loss_g)}