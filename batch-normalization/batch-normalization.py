import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    # Write code here
    x = np.asarray(x, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    beta = np.asarray(beta, dtype=float)

    if x.ndim == 2:
        mu = np.mean(x, axis=0, keepdims=True)
        var = np.mean((x - mu) ** 2, axis=0, keepdims=True)

    elif x.ndim == 4:
        mu = np.mean(x, axis=(0,2,3), keepdims=True)
        var = np.mean( (x-mu)**2, axis=(0,2,3), keepdims=True) 

        gamma = gamma.reshape(1, -1, 1, 1)
        beta = beta.reshape(1, -1, 1, 1)

    new_x = (x-mu) / (np.sqrt(var + eps))

    y = gamma*new_x + beta

    return y