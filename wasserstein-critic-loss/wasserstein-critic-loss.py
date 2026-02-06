import numpy as np

def wasserstein_critic_loss(real_scores, fake_scores):
    """
    Compute Wasserstein Critic Loss for WGAN.
    """
    # Write code here
    real_scores = np.asarray(real_scores, float)
    fake_scores = np.asarray(fake_scores, float)

    E_real = np.mean(real_scores)
    E_fake = np.mean(fake_scores)

    L = E_fake - E_real

    return L