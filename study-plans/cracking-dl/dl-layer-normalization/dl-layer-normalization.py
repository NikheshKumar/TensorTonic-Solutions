import numpy as np

def layer_normalization(x, gamma, beta, eps=1e-5, mode="forward", d_output=None):
    """
    Returns: Dict with "output", "mean", "var", "x_hat", and optionally "dx", "dgamma", "dbeta".
    """
    x = np.asarray(x, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    
    m = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_hat = (x - m) / np.sqrt(var + eps)

    y = gamma * x_hat + beta


    res = {
        "output": np.round(y, 4),
        "mean": np.round(m.squeeze(-1), 4),
        "var": np.round(var.squeeze(-1), 4),
        "x_hat": np.round(x_hat, 4),
    }

    if d_output is not None and mode=="backward":
        d_output = np.asarray(d_output, dtype=np.float64)
        dgamma = np.sum(d_output * x_hat, axis=0)
        dbeta = np.sum(d_output, axis=0)
        g = d_output * gamma
        dx = (1.0/np.sqrt(var+eps))*(g - np.mean(g, axis=-1, keepdims=True)-x_hat*np.mean(g*x_hat, axis=-1, keepdims=True))
        res.update({"dx":np.round(dx,4), "dgamma":np.round(dgamma,4), "dbeta":np.round(dbeta,4)})
    
    return res