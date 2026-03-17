import numpy as np

def nadam_step(w, m, v, grad, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Perform one Nadam update step.
    """
    # Write code here
    w = np.asarray(w, float)
    m = np.asarray(m ,float)
    v = np.asarray(v, float)
    grad = np.asarray(grad, float)


    m = beta1 * m + (1-beta1) * grad
    v = beta2 * v + (1-beta2) * (grad**2)

    num = beta1 * m + (1-beta1) * grad
    den = np.sqrt(v) + eps

    w = w - lr * num / den

    return (w, m, v)
    