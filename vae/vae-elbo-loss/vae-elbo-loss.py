import numpy as np

def vae_loss(x: np.ndarray, x_recon: np.ndarray, mu: np.ndarray, log_var: np.ndarray) -> dict:
    """
    Compute VAE ELBO loss.
    """
    # Your implementation here

    var = np.exp(log_var)

    d = -np.sum(1+log_var-var-np.square(mu), axis=1) / 2
    d_kl = np.mean(d)

    recon_feature = np.square(x-x_recon)
    recon = np.mean(recon_feature)

    total = recon + d_kl


    return {"total": total, "recon": recon, "kl":d_kl }