import numpy as np

def discriminator_loss(real_probs: np.ndarray, fake_probs: np.ndarray) -> float:
    """
    Compute discriminator loss.
    """
    # Your implementation here
    eps = 1e-8
    ld = -np.mean(np.log(real_probs+eps))-np.mean(np.log(1-fake_probs+eps))
    return ld

def generator_loss(fake_probs: np.ndarray) -> float:
    """
    Compute generator loss.
    """
    # Your implementation here
    eps = 1e-8
    lg = -np.mean(np.log(fake_probs+eps))
    return lg