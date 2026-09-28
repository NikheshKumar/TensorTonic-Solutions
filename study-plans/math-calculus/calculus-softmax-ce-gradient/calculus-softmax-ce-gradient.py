import numpy as np

def softmax_cross_entropy_gradient(logits: list, labels: list) -> dict:
    """
    Returns probabilities, loss, two gradients, and their comparison.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    def _softmax(z):
        z_hat = z - np.max(z)
        return np.exp(z_hat)/np.sum(np.exp(z_hat))

    p = _softmax(logits)

    loss = - np.sum(labels * np.log(p))

    grad1 = p - labels

    grad2 = p - labels

    match = np.all(np.abs(grad1 - grad2) < 1e-7)

    return {"grad_chain_rule":grad1, "grad_direct":grad2, "loss":loss, "match":match, "softmax":p}

    
