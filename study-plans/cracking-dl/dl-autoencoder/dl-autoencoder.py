import numpy as np

def autoencoder(x: list, W_enc: list, b_enc: list, W_dec: list, b_dec: list) -> dict:
    """
    Returns a dictionary of lists: encoded, decoded.
    """
    x = np.array(x, dtype=np.float64)
    W_enc = np.array(W_enc, dtype=np.float64)
    b_enc = np.array(b_enc, dtype=np.float64)
    W_dec = np.array(W_dec, dtype=np.float64)
    b_dec = np.array(b_dec, dtype=np.float64)

    z = np.maximum(0, W_enc @ x + b_enc)

    x_hat = np.exp(W_dec @ z + b_dec) / (1.0 + np.exp(W_dec @ z + b_dec))

    return {"encoded":np.round(z,4).tolist(), "decoded":np.round(x_hat,4).tolist()}
