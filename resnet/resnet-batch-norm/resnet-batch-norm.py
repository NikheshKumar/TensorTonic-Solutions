import numpy as np

def batch_norm_block(x, W1, W2, gamma1, beta1, gamma2, beta2, mode):
    """
    Returns the normalized residual-block result and selected mode in a dictionary.
    """
    x = np.array(x, dtype=np.float64)
    W1 = np.array(W1, dtype=np.float64)
    W2 = np.array(W2, dtype=np.float64)
    gamma1 = np.array(gamma1, dtype=np.float64)
    beta1 = np.array(beta1, dtype=np.float64)
    gamma2 = np.array(gamma2, dtype=np.float64)
    beta2 = np.array(beta2, dtype=np.float64)

    def batchnorm(z, gamma, beta):
        m = np.mean(z, axis=0)
        var = np.var(z, axis=0)
        eps = 1e-5
        return gamma * (z-m)/np.sqrt(var + eps) + beta
        
    if mode=="post":
        #linear
        z = x @ W1 
        #normalization
        z = batchnorm(z, gamma1, beta1)
        #relu
        z = np.maximum(0,z)
        #linear
        z = z @ W2 
        #normalization
        z = batchnorm(z, gamma2, beta2)
        #residual
        z = z + x
        #relu
        z = np.maximum(0,z)

    if mode=="pre":
        #normalization
        z = batchnorm(x, gamma1, beta1)
        #relu
        z = np.maximum(0,z)
        #linear
        z = z @ W1 
        #normalization
        z = batchnorm(z, gamma2, beta2)
        #relu
        z = np.maximum(0,z)
        #linear
        z = z @ W2 
        #residual
        z = z + x

    return {"output": np.round(z,4).tolist(), "mode": mode}
        