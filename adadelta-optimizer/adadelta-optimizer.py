import numpy as np

def adadelta_step(w: list, grad: list, E_grad_sq: list, E_update_sq: list, rho: float = 0.9, eps: float = 1e-6) -> dict:
    """
    Returns a dictionary with new_w, new_E_grad_sq, and new_E_update_sq.
    """
    # Write code here
    w = np.array(w, dtype=np.float64)
    grad = np.array(grad, dtype=np.float64)
    E_grad_sq = np.array(E_grad_sq, dtype=np.float64)
    E_update_sq = np.array(E_update_sq, dtype=np.float64)

    E_grad_sq = rho * E_grad_sq + (1.0 - rho) * (grad**2)

    delta = - grad * np.sqrt(E_update_sq  + eps) / np.sqrt(E_grad_sq + eps)

    E_update_sq = rho * E_update_sq + (1.0 - rho) * (delta**2)

    w = w + delta

    return {"new_w":w, "new_E_grad_sq":E_grad_sq, "new_E_update_sq":E_update_sq}