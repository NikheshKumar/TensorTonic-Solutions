import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    # Write code here
    x = np.asarray(x, float)
    gamma = np.asarray(gamma, float)
    beta = np.asarray(beta, float)

    if x.ndim==2:
        mu = np.mean(x, keepdims=True, axis=0)
        var = np.var(x, keepdims=True, axis=0)

    if x.ndim==4:
        mu = np.mean(x, axis=(0,2,3), keepdims=True)
        var = np.var(x, axis=(0,2,3), keepdims=True)
        gamma = gamma.reshape(1,-1,1,1)
        beta = beta.reshape(1,-1,1,1)
    

    x_hat = (x-mu) / np.sqrt(var + eps)

    y = gamma * x_hat + beta

    return y