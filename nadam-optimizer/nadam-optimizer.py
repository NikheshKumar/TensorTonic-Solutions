import numpy as np

def nadam_step(w: list, m: list, v: list, grad: list, lr: float = 0.002, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> dict:
    """
    Returns a dictionary with new_w, new_m, and new_v.
    """
    # Write code here
    w = np.array(w, dtype=np.float64)
    m = np.array(m, dtype=np.float64)
    v = np.array(v, dtype=np.float64)
    grad = np.array(grad, dtype=np.float64)

    m_new = beta1 * m + (1.0 - beta1) * grad
    v_new = beta2 * v + (1.0 - beta2) * (grad**2)

    m_hat = beta1 * m_new + (1.0 - beta1)*grad

    w_new = w - lr * m_hat/(np.sqrt(v_new) + eps)

    return {"new_w":w_new, "new_m":m_new, "new_v":v_new}