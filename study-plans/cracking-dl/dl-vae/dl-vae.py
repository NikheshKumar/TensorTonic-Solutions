import numpy as np

def vae_forward(x: list, W_enc: list, b_enc: list, W_mu: list, b_mu: list, W_logvar: list, b_logvar: list, W_dec: list, b_dec: list, z_sample: list) -> dict:
    """
    Returns hidden, mu, log_var, z, and reconstruction as lists.
    """
    
    x = np.array(x, dtype=np.float64)
    W_enc = np.array(W_enc, dtype=np.float64)
    b_enc = np.array(b_enc, dtype=np.float64)
    W_mu = np.array(W_mu, dtype=np.float64)
    b_mu = np.array(b_mu, dtype=np.float64)
    W_logvar = np.array(W_logvar, dtype=np.float64)
    b_logvar = np.array(b_logvar, dtype=np.float64)
    W_dec = np.array(W_dec, dtype=np.float64)
    b_dec = np.array(b_dec, dtype=np.float64)
    z_sample = np.array(z_sample, dtype=np.float64)

    h = np.maximum(W_enc @ x + b_enc, 0)

    mu = W_mu @ h + b_mu

    l = W_logvar @ h + b_logvar

    z = mu + np.exp(l/2.0) * z_sample

    r = np.exp(W_dec @ z + b_dec) / (1.0 + np.exp(W_dec @ z + b_dec))


    return {"hidden":h.tolist(), "mu":mu.tolist(), "log_var":l.tolist(), "z":z.tolist(), "reconstruction":r.tolist()}

    